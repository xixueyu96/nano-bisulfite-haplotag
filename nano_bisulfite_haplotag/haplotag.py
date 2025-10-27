import os
import mmap
import pysam
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from multiprocessing import Pool
import logging

class OptimizedHaploTagger:
    """
    Base optimized haplotype tagger class.
    """
    
    def __init__(self, bam_file: str, snp_file: str, output_dir: str,
                 min_snp: int = 2, hap_fold_change: float = 2.0, num_processors: int = 4):
        self.bam_file = bam_file
        self.snp_file = snp_file
        self.output_dir = output_dir
        self.min_snp = min_snp
        self.hap_fold_change = hap_fold_change
        self.num_processors = num_processors
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_snps_by_chromosome(self) -> Dict[str, List[Tuple[int, str, str]]]:
        """Load SNPs and group by chromosome."""
        colnames_dict = {'chr': str, 'pos': int, 'ref': str, 'alt': str, 'id': str}
        snp_df = pd.read_csv(
            self.snp_file, 
            sep="\t", 
            engine='python', 
            header=None, 
            names=list(colnames_dict.keys()), 
            dtype=colnames_dict
        )
        
        # Group SNPs by chromosome
        snps_by_chr = defaultdict(list)
        for _, row in snp_df.iterrows():
            chr_name = f"chr{row['chr']}" if not row['chr'].startswith('chr') else row['chr']
            ref, alt = row['ref'], row['alt']
            snps_by_chr[chr_name].append((int(row['pos']), ref, alt))
        
        # Sort SNPs by position within each chromosome
        for chr_name in snps_by_chr:
            snps_by_chr[chr_name].sort(key=lambda x: x[0])
            
        return dict(snps_by_chr)

    def load_snps_by_chromosome_numpy(self):
        """使用 NumPy 进一步优化"""
        import numpy as np
        
        # 读取为 numpy 数组
        data = np.genfromtxt(
            self.snp_file,
            delimiter='\t',
            dtype=[('chr', 'U10'), ('pos', 'i4'), 
                ('ref', 'U10'), ('alt', 'U10'), ('id', 'U50')],
            encoding='utf-8'
        )
        
        snps_by_chr = defaultdict(list)
        
        # 向量化处理
        for record in data:
            chr_name = record['chr'] if record['chr'].startswith('chr') else f"chr{record['chr']}"
            # ref, alt = record['genotype'].split('/')
            ref, alt = record['ref'], record['alt']
            snps_by_chr[chr_name].append((int(record['pos']), ref, alt))
        
        # 排序
        for chr_name in snps_by_chr:
            snps_by_chr[chr_name].sort(key=lambda x: x[0])
        
        return dict(snps_by_chr)
    
    def _assign_tag(self, read_base: str, ref: str, snp: str, strand: str) -> str:
        """Assign haplotype tag based on read base, reference, SNP, and strand information."""
        hp_tag = "unassigned"
        
        if ref == "C":
            if snp == "T" and strand == "OT":
                hp_tag = "unassigned"
            elif strand == "OT":
                if read_base == "C" or read_base == "T":
                    hp_tag = "genome1"
                elif read_base == snp:
                    hp_tag = "genome2"
            elif strand == "OB":
                if read_base == ref:
                    hp_tag = "genome1"
                elif snp == "G":
                    if read_base == "G" or read_base == "A":
                        hp_tag = "genome2"
                elif read_base == snp:
                    hp_tag = "genome2"
        elif snp == "C":
            if ref == "T" and strand == "OT":
                hp_tag = "unassigned"
            elif strand == "OT":
                if read_base == "C" or read_base == "T":
                    hp_tag = "genome2"
                elif read_base == ref:
                    hp_tag = "genome1"
            elif strand == "OB":
                if read_base == snp:
                    hp_tag = "genome2"
                elif ref == "G":
                    if read_base == "G" or read_base == "A":
                        hp_tag = "genome1"
                elif read_base == ref:
                    hp_tag = "genome1"
        elif ref == "G":
            if snp == "A" and strand == "OB":
                hp_tag = "unassigned"
            elif strand == "OT":
                if read_base == ref:
                    hp_tag = "genome1"
                elif snp == "C":
                    if read_base == "C" or read_base == "T":
                        hp_tag = "genome2"
                elif read_base == snp:
                    hp_tag = "genome2"
            elif strand == "OB":
                if read_base == "G" or read_base == "A":
                    hp_tag = "genome1"
                elif read_base == snp:
                    hp_tag = "genome2"
        elif snp == "G":
            if ref == "A" and strand == "OB":
                hp_tag = "unassigned"
            elif strand == "OT":
                if read_base == snp:
                    hp_tag = "genome2"
                elif ref == "C":
                    if read_base == "C" or read_base == "T":
                        hp_tag = "genome1"
                elif read_base == ref:
                    hp_tag = "genome1"
            elif strand == "OB":
                if read_base == "G" or read_base == "A":
                    hp_tag = "genome2"
                elif read_base == ref:
                    hp_tag = "genome1"
        else:
            if read_base == ref:
                hp_tag = "genome1"
            elif read_base == snp:
                hp_tag = "genome2"
        
        return hp_tag
    
    def _extract_reads(self, qname_list: List[str], hap: str) -> None:
        """Extract reads for a specific haplotype and save to BAM file."""
        if not qname_list:
            logging.warning(f"No reads found for {hap}")
            return
            
        bamfile = pysam.AlignmentFile(self.bam_file, 'rb')
        name_indexed = pysam.IndexedReads(bamfile)
        name_indexed.build()
        header = bamfile.header.copy()

        outfile = os.path.basename(self.bam_file)
        output_file = os.path.join(self.output_dir, f"{outfile}.haplotag_{hap}.bam")
        out = pysam.Samfile(output_file, 'wb', header=header)

        for qname in qname_list:
            try:
                iterator = name_indexed.find(qname)
                for read in iterator:
                    out.write(read)
            except KeyError:
                continue

        out.close()
        bamfile.close()
        
        logging.info(f"Extracted {len(qname_list)} reads to {output_file}")


class MemoryMappedHaploTagger(OptimizedHaploTagger):
    """
    Further optimized version using memory mapping for very large BAM files.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bam_cache = {}
        
    def _get_cached_bam_file(self) -> pysam.AlignmentFile:
        """Get a cached BAM file handle to avoid repeated opening."""
        if 'main' not in self._bam_cache:
            self._bam_cache['main'] = pysam.AlignmentFile(self.bam_file, 'rb')
        return self._bam_cache['main']
    
    def process_chromosome_streaming(self, chr_snps_tuple: Tuple[str, List[Tuple[int, str, str]]]) -> List[Tuple[str, str]]:
        """Process chromosome using streaming approach for very large BAM files."""
        chromosome, snps_list = chr_snps_tuple
        
        if not snps_list:
            return []
        
        # Create position lookup for fast SNP finding
        snp_dict = {}
        for pos, ref, alt in snps_list:
            snp_dict[pos] = (ref, alt)
        
        positions = sorted(snp_dict.keys())

        ## TODO test these codes
        min_pos = positions[0] - 1000  # Add buffer
        max_pos = positions[-1] + 1000
        
        all_results = []
        
        try:
            bam_file = pysam.AlignmentFile(self.bam_file, 'rb')
            
            # Stream through reads in the region
            for read in bam_file.fetch(chromosome, min_pos, max_pos):
                # Check which SNPs this read overlaps
                read_start = read.reference_start
                read_end = read.reference_end
                
                if read_start is None or read_end is None:
                    continue
                
                # Find overlapping SNPs
                overlapping_snps = [pos for pos in positions if read_start < pos <= read_end]
                
                if not overlapping_snps:
                    continue
                
                # Process each overlapping SNP
                for snp_pos in overlapping_snps:
                    ref, alt = snp_dict[snp_pos]
                    snp_result = self._process_read_at_position(read, snp_pos, ref, alt)
                    if snp_result:
                        all_results.append(snp_result)
                        
        except Exception as e:
            logging.error(f"Error processing chromosome {chromosome}: {e}")
        finally:
            if 'bam_file' in locals():
                bam_file.close()
        
        return all_results
    
    def _process_read_at_position(self, read, pos: int, ref: str, alt: str) -> Optional[Tuple[str, str]]:
        """Process a single read at a specific SNP position."""
        try:
            pair = read.get_aligned_pairs(with_seq=True)
            
            # Find SNP position (convert to 0-based)
            pair_snp = [x for x in pair if x[1] == (pos - 1)]

            if not pair_snp:
                return None
            
            pair = pair_snp[0]
            
            # Skip reads with indels at SNP position
            if pair[0] is None:
                return None

            xg_tag = read.get_tag("XG")
            read_pos = pair[0]
            read_base = pair[2]

            strand = "OT" if xg_tag == "CT" else "OB"
            
            # Get true base from read sequence
            if read_base.islower():
                true_base = read.query_sequence[read_pos]
            else:
                true_base = read_base

            hp_tag = self._assign_tag(true_base, ref, alt, strand)
            qname = read.query_name

            return (qname, hp_tag)
            
        except Exception as e:
            logging.debug(f"Error processing read {read.query_name} at position {pos}: {e}")
            return None
    
    def analyze_snps_streaming(self) -> Dict[str, int]:
        """
        Main analysis function using streaming approach.
        
        Returns:
            Dictionary with analysis statistics
        """
        logging.info("Starting streaming haplotype analysis...")
        
        # Load SNPs grouped by chromosome
        snps_by_chr = self.load_snps_by_chromosome_numpy()
        total_snps = sum(len(snps) for snps in snps_by_chr.values())
        logging.info(f"Loaded {total_snps} SNPs across {len(snps_by_chr)} chromosomes")
        
        # Process chromosomes in parallel using streaming
        chr_snps_list = list(snps_by_chr.items())
        
        all_results = []
        with Pool(processes=self.num_processors) as pool:
            results = pool.map(self.process_chromosome_streaming, chr_snps_list)
            for result in results:
                all_results.extend(result)
        
        logging.info(f"Processed {len(all_results)} read-SNP pairs")
        
        if not all_results:
            logging.warning("No results found")
            return {'total_snps': total_snps, 'reads_with_snps': 0, 'hap1_reads': 0, 'hap2_reads': 0}
        
        # Convert to DataFrame for read-level analysis
        df = pd.DataFrame(all_results, columns=['qname', 'haplotype'])
        
        # Count haplotype assignments per read
        hap_counts = df.groupby('qname')['haplotype'].value_counts().unstack(fill_value=0)
        
        # Ensure required columns exist
        for col in ['genome1', 'genome2', 'unassigned']:
            if col not in hap_counts.columns:
                hap_counts[col] = 0
        
        # Apply haplotype assignment rules
        conditions = [
            (hap_counts['genome1'] >= self.min_snp) & 
            ((hap_counts['genome2'] == 0) | (hap_counts['genome1'] >= self.hap_fold_change * hap_counts['genome2'])),
            (hap_counts['genome2'] >= self.min_snp) & 
            ((hap_counts['genome1'] == 0) | (hap_counts['genome2'] >= self.hap_fold_change * hap_counts['genome1']))
        ]
        
        choices = ['genome1', 'genome2']
        hap_counts['final_haplotype'] = np.select(conditions, choices, default=np.nan)
        
        # Get final assignments
        hap1_reads = list(hap_counts[hap_counts['final_haplotype'] == 'genome1'].index)
        hap2_reads = list(hap_counts[hap_counts['final_haplotype'] == 'genome2'].index)
        
        # Extract reads to separate BAM files
        self._extract_reads(hap1_reads, "hap1")
        self._extract_reads(hap2_reads, "hap2")
        
        # Generate statistics
        stats = {
            'total_snps': total_snps,
            'reads_with_snps': len(hap_counts),
            'hap1_reads': len(hap1_reads),
            'hap2_reads': len(hap2_reads),
            'tagged_reads': len(hap1_reads) + len(hap2_reads)
        }
        
        logging.info("Streaming analysis complete!")
        logging.info(f"HAP1 reads: {len(hap1_reads)}")
        logging.info(f"HAP2 reads: {len(hap2_reads)}")
        
        return stats
    
    def cleanup(self):
        """Clean up cached resources."""
        for bam_file in self._bam_cache.values():
            if bam_file:
                bam_file.close()
        self._bam_cache.clear()