#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
import argparse

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("App")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plant Disease Data Collector Tool")
    parser.add_argument("--source-dir", type=str, help="Source directory for local import")
    parser.add_argument("--output-dir", type=str, help="Output directory for dataset")
    parser.add_argument("--search-terms", type=str, help="Comma-separated search terms for web collection")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--images-per-class", type=int, default=100, help="Images to collect per class in web mode")
    parser.add_argument("--sources", type=str, default="google,bing,baidu", help="Comma-separated web sources")
    parser.add_argument("--split-ratio", type=str, default="0.8,0.1,0.1", help="Split ratio for local import, e.g. 0.8,0.1,0.1")
    parser.add_argument("--enable-size-filter", action="store_true", help="Enable size filtering")
    parser.add_argument("--min-width", type=int, default=224, help="Minimum image width")
    parser.add_argument("--min-height", type=int, default=224, help="Minimum image height")
    parser.add_argument("--quality-filter", action="store_true", help="Enable low-quality filtering for local import")
    parser.add_argument("--no-quality-filter", action="store_true", help="Disable low-quality filtering")
    parser.add_argument("--deduplicate", action="store_true", help="Enable duplicate removal for local import")
    parser.add_argument("--no-deduplicate", action="store_true", help="Disable duplicate removal")
    parser.add_argument("--min-file-size-kb", type=int, default=12, help="Minimum file size for quality filtering")
    parser.add_argument("--min-variance", type=float, default=25.0, help="Minimum grayscale variance for quality filtering")
    parser.add_argument("--max-aspect-ratio", type=float, default=6.0, help="Maximum image aspect ratio for quality filtering")
    parser.add_argument("--generate-manifest", action="store_true", help="Write a dataset manifest JSON next to the output")
    parser.add_argument("--manifest-name", type=str, default="manifest.json", help="Manifest file name")
    return parser.parse_args()


def parse_split_ratio(raw_ratio: str):
    """Convert a comma-separated ratio string into a split dict."""
    parts = [part.strip() for part in raw_ratio.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--split-ratio must have exactly three comma-separated values")

    train, val, test = [float(part) for part in parts]
    return {"train": train, "val": val, "test": test}


def resolve_bool(explicit_on: bool, explicit_off: bool, default: bool) -> bool:
    """Resolve a tri-state boolean from CLI flags."""
    if explicit_on and explicit_off:
        raise ValueError("Conflicting boolean flags were provided")
    if explicit_on:
        return True
    if explicit_off:
        return False
    return default


def build_quality_config(args):
    """Build quality filter config from CLI flags."""
    return {
        "min_file_size_kb": args.min_file_size_kb,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "min_variance": args.min_variance,
        "max_aspect_ratio": args.max_aspect_ratio,
    }


def import_dataset_maker():
    """Import DatasetMaker with packaged and direct-script fallbacks."""
    try:
        from .dataset_maker import DatasetMaker
    except ImportError:
        from tools.dataset_collector.dataset_maker import DatasetMaker
    return DatasetMaker


def run_gui(args, logger):
    """Launch the GUI only when explicitly needed."""
    try:
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            print("PyQt6 is not installed. Please install it using:")
            print("pip install PyQt6>=6.4.0")
            return 1

        try:
            from .dataset_collector import MainWindow
        except ImportError:
            from tools.dataset_collector.dataset_collector import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("Plant Disease Data Collector")
        app.setStyle("Fusion")

        window = MainWindow()

        if args.source_dir:
            window.importer_tab.source_dir_edit.setText(args.source_dir)
            window.importer_tab.scan_source_directory()

        if args.output_dir:
            window.importer_tab.output_dir_edit.setText(args.output_dir)
            window.scraper_tab.output_dir_edit.setText(args.output_dir)

        if args.search_terms:
            window.scraper_tab.search_terms_edit.setPlainText(args.search_terms.replace(",", "\n"))
            window.tab_widget.setCurrentIndex(1)

        window.show()
        return app.exec()

    except Exception as e:
        logger.error(f"Error starting GUI: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error starting GUI: {str(e)}")
        return 1


def run_headless(args, logger):
    """Run dataset preparation in headless mode."""
    DatasetMaker = import_dataset_maker()
    dataset_maker = DatasetMaker()
    output_dir = args.output_dir or os.path.join(parent_dir, "data")
    manifest_name = args.manifest_name if args.generate_manifest else None
    size_filter = None

    if args.enable_size_filter:
        size_filter = {
            "enabled": True,
            "min_width": args.min_width,
            "min_height": args.min_height,
        }

    quality_config = build_quality_config(args)

    if args.search_terms:
        search_terms = [term.strip() for term in args.search_terms.split(",") if term.strip()]
        if not search_terms:
            raise ValueError("No valid search terms were provided")

        sources = [source.strip() for source in args.sources.split(",") if source.strip()]
        quality_filter = resolve_bool(args.quality_filter, args.no_quality_filter, True)
        deduplicate = resolve_bool(args.deduplicate, args.no_deduplicate, True)

        logger.info(f"Collecting data for search terms: {search_terms}")
        result = dataset_maker.make_from_web(
            search_terms=search_terms,
            output_dir=output_dir,
            images_per_class=args.images_per_class,
            sources=sources,
            generate_labels=True,
            quality_filter=quality_filter,
            deduplicate=deduplicate,
            size_filter=size_filter,
            quality_config=quality_config,
            manifest_name=manifest_name,
        )
        return 0 if result else 1

    if args.source_dir:
        split_ratio = parse_split_ratio(args.split_ratio)
        quality_filter = resolve_bool(args.quality_filter, args.no_quality_filter, False)
        deduplicate = resolve_bool(args.deduplicate, args.no_deduplicate, False)

        logger.info(f"Importing data from: {args.source_dir}")
        result = dataset_maker.make_from_local(
            source_dir=args.source_dir,
            output_dir=output_dir,
            split_ratio=split_ratio,
            generate_labels=True,
            size_filter=size_filter,
            quality_filter=quality_filter,
            deduplicate=deduplicate,
            quality_config=quality_config,
            manifest_name=manifest_name,
        )
        return 0 if result else 1

    logger.error("Headless mode requires --source-dir or --search-terms")
    print("Error: Headless mode requires --source-dir or --search-terms")
    return 1


def main():
    """Main application entry point."""
    args = parse_arguments()

    logger = setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Plant Disease Data Collector Tool")

    try:
        if args.headless:
            logger.info("Running in headless mode")
            return run_headless(args, logger)
        return run_gui(args, logger)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
