"""
Command-line interface for SCENIC+ and PINNACLE integration.

This module provides a command-line interface for running the integration
analysis without writing Python code.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .core import ScenicPinnacleIntegrator


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: Optional[str]) -> Dict:
    """Load configuration from JSON file."""
    if not config_path:
        return {}
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def run_integration(args) -> None:
    """Run the integration analysis."""
    print("SCENIC+ and PINNACLE Integration")
    print("=" * 40)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize integrator
    print("Initializing integrator...")
    integrator = ScenicPinnacleIntegrator(config=config)
    
    # Load data
    print(f"Loading SCENIC+ data from {args.scenic_data}...")
    integrator.load_scenic_data(args.scenic_data, data_format=args.scenic_format)
    
    print(f"Loading PINNACLE data from {args.pinnacle_data}...")
    integrator.load_pinnacle_data(args.pinnacle_data, data_format=args.pinnacle_format)
    
    # Run analysis
    if args.workflow == 'complete':
        print("Running complete workflow...")
        conditions = args.conditions.split(',') if args.conditions else None
        comparisons = None
        
        if args.comparisons:
            comparisons = []
            for comp in args.comparisons.split(';'):
                cond1, cond2 = comp.split(',')
                comparisons.append((cond1.strip(), cond2.strip()))
        
        summary = integrator.run_complete_workflow(
            scenic_path=args.scenic_data,
            pinnacle_path=args.pinnacle_data,
            output_dir=Path(args.output),
            conditions=conditions,
            comparisons=comparisons
        )
        
        print("\nWorkflow Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    else:
        # Step-by-step workflow
        print("Integrating networks...")
        integrated_networks = integrator.integrate_networks()
        
        if args.differential and len(integrated_networks) >= 2:
            conditions = list(integrated_networks.keys())
            print(f"Performing differential analysis: {conditions[0]} vs {conditions[1]}...")
            diff_results = integrator.differential_analysis(
                conditions[0], conditions[1], analysis_type='both'
            )
            
            # Print summary
            summary = diff_results.get('summary', {})
            for analysis_type, stats in summary.items():
                print(f"  {analysis_type.title()} changes:")
                for stat_name, value in stats.items():
                    print(f"    {stat_name}: {value}")
        
        if args.visualize:
            print("Generating visualizations...")
            for condition in list(integrated_networks.keys())[:3]:  # Limit to first 3
                viz_dir = Path(args.output) / 'visualizations' / condition
                integrator.visualize_networks(condition, viz_dir)
        
        print("Exporting results...")
        integrator.export_results(Path(args.output) / 'results')
    
    print(f"\nAnalysis completed! Results saved to: {args.output}")


def run_validate(args) -> None:
    """Validate input data files."""
    print("Validating input data...")
    
    from .utils import QualityController
    from .data_processing import ScenicProcessor, PinnacleProcessor
    
    qc = QualityController()
    
    # Validate SCENIC+ data
    if args.scenic_data:
        print(f"Validating SCENIC+ data: {args.scenic_data}")
        scenic_processor = ScenicProcessor()
        scenic_networks = scenic_processor.load_data(args.scenic_data, args.scenic_format)
        validated_scenic = qc.validate_scenic_networks(scenic_networks)
        
        print(f"  Original networks: {len(scenic_networks)}")
        print(f"  Validated networks: {len(validated_scenic)}")
        
        if len(validated_scenic) < len(scenic_networks):
            print("  Warning: Some networks failed validation")
    
    # Validate PINNACLE data
    if args.pinnacle_data:
        print(f"Validating PINNACLE data: {args.pinnacle_data}")
        pinnacle_processor = PinnacleProcessor()
        pinnacle_embeddings = pinnacle_processor.load_data(args.pinnacle_data, args.pinnacle_format)
        validated_pinnacle = qc.validate_pinnacle_embeddings(pinnacle_embeddings)
        
        print(f"  Original embeddings: {len(pinnacle_embeddings)}")
        print(f"  Validated embeddings: {len(validated_pinnacle)}")
        
        if len(validated_pinnacle) < len(pinnacle_embeddings):
            print("  Warning: Some embeddings failed validation")
    
    print("Validation completed!")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="SCENIC+ and PINNACLE Integration Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete integration workflow
  scenic-pinnacle integrate --scenic-data scenic.pkl --pinnacle-data pinnacle.pkl --output results/
  
  # Run with custom configuration
  scenic-pinnacle integrate --config config.json --scenic-data scenic.csv --pinnacle-data pinnacle.npz --output results/
  
  # Validate input data
  scenic-pinnacle validate --scenic-data scenic.pkl --pinnacle-data pinnacle.pkl
  
  # Run with specific conditions and comparisons
  scenic-pinnacle integrate --scenic-data scenic.pkl --pinnacle-data pinnacle.pkl \\
    --conditions "healthy,disease,treatment" --comparisons "healthy,disease;disease,treatment" \\
    --output results/
        """
    )
    
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Integration command
    integrate_parser = subparsers.add_parser('integrate', help='Run integration analysis')
    integrate_parser.add_argument('--scenic-data', required=True, 
                                help='Path to SCENIC+ data file')
    integrate_parser.add_argument('--pinnacle-data', required=True,
                                help='Path to PINNACLE data file')
    integrate_parser.add_argument('--output', '-o', required=True,
                                help='Output directory for results')
    integrate_parser.add_argument('--config', '-c',
                                help='Path to configuration JSON file')
    integrate_parser.add_argument('--scenic-format', default='pickle',
                                choices=['pickle', 'csv', 'h5ad'],
                                help='Format of SCENIC+ data (default: pickle)')
    integrate_parser.add_argument('--pinnacle-format', default='pickle',
                                choices=['pickle', 'csv', 'npz'],
                                help='Format of PINNACLE data (default: pickle)')
    integrate_parser.add_argument('--workflow', default='complete',
                                choices=['complete', 'stepwise'],
                                help='Workflow type (default: complete)')
    integrate_parser.add_argument('--conditions',
                                help='Comma-separated list of conditions to analyze')
    integrate_parser.add_argument('--comparisons',
                                help='Semicolon-separated list of condition pairs (e.g., "cond1,cond2;cond2,cond3")')
    integrate_parser.add_argument('--differential', action='store_true',
                                help='Perform differential analysis')
    integrate_parser.add_argument('--visualize', action='store_true',
                                help='Generate visualizations')
    integrate_parser.set_defaults(func=run_integration)
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Validate input data')
    validate_parser.add_argument('--scenic-data',
                               help='Path to SCENIC+ data file')
    validate_parser.add_argument('--pinnacle-data',
                               help='Path to PINNACLE data file')
    validate_parser.add_argument('--scenic-format', default='pickle',
                               choices=['pickle', 'csv', 'h5ad'],
                               help='Format of SCENIC+ data (default: pickle)')
    validate_parser.add_argument('--pinnacle-format', default='pickle',
                               choices=['pickle', 'csv', 'npz'],
                               help='Format of PINNACLE data (default: pickle)')
    validate_parser.set_defaults(func=run_validate)
    
    return parser


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up logging
    setup_logging(args.verbose)
    
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()

