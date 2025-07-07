# Planet Nine Large-Scale Search System

## Overview

This system extends the validated Planet Nine detection pipeline to process 50-100 square degrees of sky with advanced batch processing, enhanced candidate analysis, and pattern detection capabilities.

## System Components

### 1. Large-Scale Search Manager (`large_scale_search.py`)

**Purpose**: Orchestrate searches across multiple sky regions with parallel processing

**Key Features**:
- Automatic generation of high-priority search regions based on theoretical predictions
- Parallel processing of multiple sky regions using ProcessPoolExecutor
- SQLite database for tracking progress and storing results
- Intelligent region selection (anti-clustering zones, perihelion approach areas, high galactic latitudes)
- Comprehensive error handling and recovery mechanisms

**Usage**:
```bash
# Process 100 square degrees with 8 parallel workers
python large_scale_search.py --area 100 --workers 8

# Resume from previous interrupted search
python large_scale_search.py --area 100 --workers 4 --resume
```

**Output**:
- Search progress database with all candidates and regions
- JSON summary with detection statistics and patterns
- Individual region processing results

### 2. Enhanced Candidate Ranking (`enhanced_candidate_ranking.py`)

**Purpose**: Apply sophisticated ranking algorithms to identify the most promising Planet Nine candidates

**Ranking Criteria** (weighted combination):
- **Motion Score (25%)**: Prefers Planet Nine range (0.2-0.8 arcsec/year)
- **Quality Score (20%)**: Detection algorithm confidence
- **Novelty Score (20%)**: How unusual the detection is (faint, isolated, consistent flux)
- **Validation Score (15%)**: Distance from known astronomical objects
- **Consistency Score (10%)**: Multi-epoch reliability
- **Theoretical Score (10%)**: Location matches orbital predictions

**Tier Classification**:
- **Tier 1 (Exceptional)**: Top 5% - highest priority for follow-up
- **Tier 2 (High Priority)**: Top 15% - immediate attention required
- **Tier 3 (Moderate)**: Top 30% - standard follow-up queue
- **Tier 4 (Low Priority)**: Remaining candidates

**Usage**:
```bash
# Rank candidates from search database
python enhanced_candidate_ranking.py --database results/large_scale_search/search_progress.db --min-quality 0.3
```

**Output**:
- Ranked candidate CSV file with all scores
- JSON report with tier analysis and top candidates
- Visualization plots showing ranking distributions

### 3. Progress Tracking & Recovery (`progress_tracking.py`)

**Purpose**: Monitor long-running searches with real-time metrics and recovery capabilities

**Features**:
- Real-time system monitoring (CPU, memory, disk usage)
- Periodic checkpointing for crash recovery
- Progress database with detailed event logging
- Resource usage warnings and optimization suggestions
- Graceful shutdown handling with signal management

**Integration**:
```python
from progress_tracking import progress_tracking

with progress_tracking("search_id") as tracker:
    for region in regions:
        tracker.update_region_start(region.region_id)
        # ... process region ...
        tracker.update_region_complete(region.region_id, candidates, time)
```

**Recovery**:
```python
from progress_tracking import RecoveryManager

recovery = RecoveryManager("search_id")
if recovery.can_recover():
    remaining_regions = recovery.recover_region_list(original_regions)
```

### 4. Pattern Detection (`pattern_detection.py`)

**Purpose**: Identify systematic trends and anomalies across multiple search regions

**Pattern Types Detected**:

**Spatial Patterns**:
- Clustering of candidates in specific sky regions
- Density variations across different areas
- Correlation with galactic coordinates

**Motion Patterns**:
- Multi-modal motion distributions
- Excess candidates in Planet Nine range
- Systematic motion direction biases

**Temporal Patterns**:
- Time-dependent detection rates
- Processing efficiency variations
- Systematic effects over observation periods

**Anomaly Detection**:
- Ultra-slow motion candidates (<0.05 arcsec/year)
- Exceptionally high-quality detections
- Very faint object populations

**Usage**:
```bash
# Analyze patterns from completed search
python pattern_detection.py --database results/large_scale_search/search_progress.db --min-candidates 10
```

**Output**:
- Pattern analysis JSON with statistical significance
- Cross-correlation analysis between variables
- Recommendations for algorithm improvements
- Visualization plots of detected patterns

## Workflow Example

### Complete Large-Scale Search Pipeline

```bash
# 1. Run large-scale search (50-100 square degrees)
python large_scale_search.py --area 75 --workers 6

# 2. Rank all candidates for follow-up prioritization
python enhanced_candidate_ranking.py --database results/large_scale_search/search_progress.db

# 3. Detect patterns for systematic effect analysis
python pattern_detection.py --database results/large_scale_search/search_progress.db

# 4. Run demonstration (quick test of all components)
python run_large_scale_demo.py
```

## Database Schema

The system uses SQLite databases to track progress and store results:

### Search Progress Database
- **search_regions**: Region definitions and processing status
- **candidate_detections**: All detected candidates with properties
- **search_patterns**: Identified patterns across regions
- **system_metrics**: Performance monitoring data
- **error_log**: Processing errors and recovery attempts

### Key Tables

**search_regions**:
```sql
region_id, ra_center, dec_center, width, height, priority, 
theoretical_basis, status, processing_time, total_candidates,
high_quality_candidates, planet_nine_candidates
```

**candidate_detections**:
```sql
detection_id, region_id, ra, dec, motion_arcsec_year, 
quality_score, start_flux, is_planet_nine_candidate, 
validation_status
```

## Performance Characteristics

### Scalability
- **Processing Rate**: ~10-15 regions per hour (depends on hardware)
- **Memory Usage**: ~2-4 GB per worker process
- **Disk Requirements**: ~100 MB per square degree processed
- **Network**: ~500 MB download per square degree

### Optimization Features
- Parallel processing with configurable worker count
- Incremental checkpointing for crash recovery
- Memory-efficient streaming of large datasets
- Automated cleanup of temporary files
- Progress-based resource allocation

## Integration with Existing Pipeline

The large-scale system builds on the validated single-region pipeline:

1. **Data Download**: Uses existing `survey_downloader.py`
2. **Image Processing**: Leverages `image_alignment.py` and processing modules
3. **Detection**: Applies proven `moving_object_detection.py` algorithms
4. **Validation**: Extends `candidate_validation.py` with enhanced cross-matching

## Configuration

### High-Priority Search Regions

The system automatically generates regions based on:

1. **Anti-clustering Zones**: Opposite to known KBO perihelia
2. **Perihelion Approach**: Where Planet Nine would be brightest
3. **High Galactic Latitudes**: Reduced stellar contamination
4. **Systematic Grid**: Complete coverage of remaining area

### Customization Options

```python
# Custom search criteria
criteria = RankingCriteria(
    motion_weight=0.3,      # Emphasize motion range
    novelty_weight=0.25,    # Prioritize unusual detections
    theoretical_weight=0.15 # Weight theoretical predictions
)

# Custom region generation
regions = manager.generate_priority_regions(
    total_area_sq_deg=50,
    focus_anti_clustering=True,
    avoid_galactic_plane=True
)
```

## Results and Output

### Summary Statistics
- Total candidates detected across all regions
- Regional efficiency and detection rate analysis
- Pattern significance and recommendations
- Processing performance metrics

### Follow-up Priorities
- Tier 1 candidates for immediate spectroscopic follow-up
- Tier 2 candidates for photometric monitoring
- Regional patterns requiring investigation
- Algorithm improvements suggested

### Data Products
- Comprehensive candidate catalogs with enhanced properties
- Cross-validated detection databases
- Pattern analysis reports with statistical significance
- Performance monitoring and optimization recommendations

## Error Handling and Recovery

### Robust Error Management
- Individual region failures don't stop the entire search
- Automatic retry mechanisms for transient failures
- Comprehensive error logging with stack traces
- Graceful degradation for partial processing failures

### Recovery Capabilities
- Automatic checkpoint saving every 5 minutes
- Resume capability from any checkpoint
- Progress preservation across system restarts
- Partial result recovery from failed searches

## Future Enhancements

### Planned Improvements
1. **Real-time Gaia cross-matching** during detection
2. **Machine learning classification** for stellar vs TNO separation
3. **Automated spectroscopic follow-up** triggers
4. **LSST integration** for deeper survey data
5. **Distributed processing** across multiple machines

### Scalability Roadmap
- **Phase 1**: 100 square degrees (current capability)
- **Phase 2**: 1000 square degrees with distributed processing
- **Phase 3**: All-sky survey integration with LSST
- **Phase 4**: Real-time discovery pipeline with automated follow-up

---

*This large-scale search system represents a significant advancement in automated Planet Nine detection, providing the tools necessary to systematically search meaningful areas of sky with comprehensive analysis and validation.*