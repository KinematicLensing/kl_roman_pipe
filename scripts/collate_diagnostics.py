#!/usr/bin/env python3

"""
Collate diagnostic images from the Roman KL Pipe repo tests.

Auto-discovers PNG files in tests/out/*/, generates HTML and PDF reports
with git metadata, test descriptions, and image metadata.

Usage
-----
    python scripts/collate_diagnostics.py --html --open
    python scripts/collate_diagnostics.py --pdf
    python scripts/collate_diagnostics.py --html --pdf --sync
    python scripts/collate_diagnostics.py --html --sync --no-confirm

Makefile targets
----------------
    make show-diagnostics        # HTML + open in browser
    make diagnostics-pdf         # PDF only
    make sync-diagnostics        # HTML + PDF + rsync to rigel
    make sync-diagnostics-html   # HTML only + rsync to rigel
    make test-show               # Run tests + show HTML
    make test-sync               # Run tests + sync diagnostics

Examples
--------
    # Generate HTML report and open in browser
    $ python scripts/collate_diagnostics.py --html --open

    # Generate PDF report only
    $ python scripts/collate_diagnostics.py --pdf

    # Generate both and sync to rigel (with confirmation)
    $ python scripts/collate_diagnostics.py --html --pdf --sync

    # Sync without confirmation prompt
    $ python scripts/collate_diagnostics.py --html --pdf --sync --no-confirm
"""

import argparse
import base64
import datetime
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# remote sync configuration
# NOTE/TODO: Add a remote host, if desired
# REMOTE_HOST = 'user@host'
# REMOTE_DIR = '/path/to/remote/diagnostics'

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields: html, pdf, open, sync, no_confirm.
    """
    parser = argparse.ArgumentParser(
        description='Collate diagnostic images from Roman KL Pipe tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML report',
    )
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Generate PDF report',
    )
    parser.add_argument(
        '--open',
        action='store_true',
        help='Open HTML report in browser (requires --html)',
    )
    parser.add_argument(
        '--sync',
        action='store_true',
        help='Sync generated reports to remote server via rsync (requires REMOTE_HOST and REMOTE_DIR to be set)',
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt before rsync (use with --sync)',
    )
    return parser.parse_args()


def get_git_metadata() -> Dict[str, any]:
    """
    Extract git repository metadata.

    Returns
    -------
    dict
        Git metadata with keys:
        - branch: str (current branch name)
        - commit_hash: str (short 7-char hash)
        - commit_hash_long: str (full 40-char hash)
        - commit_message: str (latest commit message)
        - dirty: bool (True if tracked files modified or staged)
        - dirty_files: list[str] (modified/staged files)

    Notes
    -----
    Only checks tracked files for dirty status (ignores untracked files).
    Returns 'unknown' for fields if git commands fail.
    """
    repo_root = Path(__file__).parent.parent

    def run_git_cmd(cmd: List[str]) -> str:
        """Run git command and return stdout."""
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'

    # get branch
    branch = run_git_cmd(['git', 'branch', '--show-current'])

    # get commit hash (short and long)
    commit_hash_long = run_git_cmd(['git', 'rev-parse', 'HEAD'])
    commit_hash = run_git_cmd(['git', 'rev-parse', '--short=7', 'HEAD'])

    # get commit message
    commit_msg = run_git_cmd(['git', 'log', '-1', '--format=%s'])

    # check for uncommitted changes to TRACKED files only (ignore untracked)
    diff_unstaged = run_git_cmd(['git', 'diff', '--name-only'])
    diff_staged = run_git_cmd(['git', 'diff', '--cached', '--name-only'])

    dirty = bool(diff_unstaged) or bool(diff_staged)
    dirty_files = []
    if diff_unstaged and diff_unstaged != 'unknown':
        dirty_files.extend(diff_unstaged.split('\n'))
    if diff_staged and diff_staged != 'unknown':
        dirty_files.extend(
            ['(staged) ' + f for f in diff_staged.split('\n') if f]
        )

    return {
        'branch': branch,
        'commit_hash': commit_hash,
        'commit_hash_long': commit_hash_long,
        'commit_message': commit_msg,
        'dirty': dirty,
        'dirty_files': [f for f in dirty_files if f],
    }


def discover_diagnostic_images(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover PNG files in tests/out/*/ grouped by test category.

    Parameters
    ----------
    base_dir : Path
        Base directory to search (typically tests/out/).

    Returns
    -------
    dict
        Mapping of test_category -> list[Path], sorted alphabetically.
        Example: {'interface': [Path(...), ...], 'validation': [...]}

    Notes
    -----
    Only discovers PNG files. FITS files are skipped.
    Test category is determined by subdirectory name.
    """
    if not base_dir.exists():
        return {}

    images_by_test = {}

    # recursively find all PNG files
    for png_file in base_dir.glob('**/*.png'):
        # get test category from parent directory
        # tests/out/interface/file.png -> 'interface'
        relative = png_file.relative_to(base_dir)

        # skip if in diagnostics output directory
        if relative.parts[0] == 'diagnostics':
            continue

        test_category = relative.parts[0]

        if test_category not in images_by_test:
            images_by_test[test_category] = []

        images_by_test[test_category].append(png_file)

    # sort files within each category alphabetically
    for test_category in images_by_test:
        images_by_test[test_category].sort()

    return images_by_test


def get_image_metadata(image_path: Path) -> Dict[str, any]:
    """
    Extract metadata for a single image file.

    Parameters
    ----------
    image_path : Path
        Path to PNG image file.

    Returns
    -------
    dict
        Image metadata with keys:
        - filename: str
        - file_size: str (human-readable, e.g., '3.2 MB')
        - modification_time: str (ISO format)
        - width: int (pixels)
        - height: int (pixels)
        - dimensions: str (formatted as 'WxH')
    """
    # get file size
    size_bytes = image_path.stat().st_size
    if size_bytes < 1024:
        file_size = f'{size_bytes} B'
    elif size_bytes < 1024**2:
        file_size = f'{size_bytes / 1024:.1f} KB'
    else:
        file_size = f'{size_bytes / (1024**2):.1f} MB'

    # get modification time
    mod_time = datetime.datetime.fromtimestamp(image_path.stat().st_mtime)
    mod_time_str = mod_time.strftime('%Y-%m-%d %H:%M:%S')

    # get image dimensions
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        width, height = 0, 0

    return {
        'filename': image_path.name,
        'file_size': file_size,
        'modification_time': mod_time_str,
        'width': width,
        'height': height,
        'dimensions': f'{width}x{height}',
    }


def get_test_descriptions() -> Dict[str, str]:
    """
    Return hardcoded test category descriptions.

    Returns
    -------
    dict
        Mapping of test_category -> description.
    """
    return {
        'intensity': 'Unit tests for intensity models: model instantiation, parameter conversion, evaluation in different planes, and physical properties',
        'velocity': 'Unit tests for velocity models: rotation curve parameterization, circular velocity evaluation, and coordinate transformations',
        'likelihood_slices': 'Parameter recovery tests using likelihood slicing to validate forward models and verify parameters can be recovered in principle',
        'optimizer_recovery': 'Parameter recovery tests using gradient-based optimization with JAX autodiff to validate inference in realistic scenarios',
        'tng_diagnostics': 'TNG50 data vector generation diagnostics: rendering quality, coordinate transforms, gridding algorithms, and orientation sweeps',
        'tng_exploration': 'TNG50 exploration and analysis: galaxy property investigations and systematic studies',
    }


def generate_html_report(
    images_by_test: Dict[str, List[Path]],
    git_metadata: Dict[str, any],
    output_path: Path,
) -> None:
    """
    Generate self-contained HTML report with embedded images.

    Parameters
    ----------
    images_by_test : dict
        Mapping of test_category -> list[Path].
    git_metadata : dict
        Git repository metadata from get_git_metadata().
    output_path : Path
        Path to write HTML file.

    Notes
    -----
    Images are base64-encoded and embedded directly in HTML.
    Uses HTML5 <details> elements for collapsible sections.
    """
    test_descriptions = get_test_descriptions()

    # build HTML content
    html_parts = []

    # html header
    html_parts.append(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPHEREx Diagnostic Images</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        .git-metadata {
            background: white;
            border-left: 4px solid #667eea;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .git-metadata h2 {
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: #667eea;
        }
        .git-metadata ul {
            list-style: none;
        }
        .git-metadata li {
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }
        .git-metadata li:last-child {
            border-bottom: none;
        }
        .git-metadata strong {
            display: inline-block;
            width: 150px;
            color: #555;
        }
        .status-clean {
            color: #22c55e;
            font-weight: bold;
        }
        .status-dirty {
            color: #ef4444;
            font-weight: bold;
        }
        details {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        details[open] {
            border-color: #667eea;
        }
        summary {
            font-size: 1.5rem;
            font-weight: 600;
            cursor: pointer;
            padding: 0.5rem;
            color: #667eea;
            user-select: none;
        }
        summary:hover {
            color: #764ba2;
        }
        .test-description {
            font-style: italic;
            color: #666;
            margin: 1rem 0 1.5rem 0;
            padding: 0.5rem;
            background: #f9f9f9;
            border-left: 3px solid #667eea;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-top: 1rem;
        }
        .image-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .image-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .diagnostic-image {
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
            background: #fafafa;
        }
        .image-metadata {
            padding: 1rem;
            background: #fafafa;
        }
        .image-metadata p {
            margin: 0.3rem 0;
            font-size: 0.9rem;
            color: #555;
        }
        .image-metadata .filename {
            font-weight: 600;
            color: #333;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        .footer {
            text-align: center;
            padding: 2rem;
            color: #888;
            font-size: 0.9rem;
        }
    </style>
    <script>
        // click to open image in new tab
        function openImage(src) {
            window.open(src, '_blank');
        }
    </script>
</head>
<body>
    <div class="container">
"""
    )

    # header with timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_parts.append(
        f"""
        <div class="header">
            <h1>Roman KL Pipeline Diagnostics</h1>
            <p>Generated: {timestamp}</p>
        </div>
"""
    )

    # git metadata section
    dirty_status = 'DIRTY' if git_metadata['dirty'] else 'CLEAN'
    status_class = 'status-dirty' if git_metadata['dirty'] else 'status-clean'

    dirty_files_html = ''
    if git_metadata['dirty'] and git_metadata['dirty_files']:
        dirty_files_html = '<ul style="margin-left: 2rem; margin-top: 0.5rem; font-size: 0.85rem;">'
        for f in git_metadata['dirty_files'][:10]:  # limit to 10 files
            dirty_files_html += f'<li>{f}</li>'
        if len(git_metadata['dirty_files']) > 10:
            remaining = len(git_metadata['dirty_files']) - 10
            dirty_files_html += (
                f'<li><em>... and {remaining} more files</em></li>'
            )
        dirty_files_html += '</ul>'

    html_parts.append(
        f"""
        <div class="git-metadata">
            <h2>Git Repository Status</h2>
            <ul>
                <li><strong>Branch:</strong> {git_metadata['branch']}</li>
                <li><strong>Commit:</strong> {git_metadata['commit_hash']} ({git_metadata['commit_message']})</li>
                <li><strong>Status:</strong> <span class="{status_class}">{dirty_status}</span></li>
            </ul>
            {dirty_files_html}
        </div>
"""
    )

    # test sections
    for test_category in sorted(images_by_test.keys()):
        image_paths = images_by_test[test_category]
        description = test_descriptions.get(
            test_category,
            f'Diagnostic images from {test_category} test',
        )

        html_parts.append(
            f"""
        <details open>
            <summary>{test_category.replace('_', ' ').title()} ({len(image_paths)} images)</summary>
            <p class="test-description">{description}</p>
            <div class="image-grid">
"""
        )

        # add images
        for img_path in image_paths:
            metadata = get_image_metadata(img_path)

            # encode image as base64
            try:
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    img_src = f'data:image/png;base64,{img_data}'
            except Exception as e:
                print(f'Warning: Could not encode {img_path}: {e}')
                continue

            html_parts.append(
                f"""
                <div class="image-card">
                    <img src="{img_src}" class="diagnostic-image" alt="{metadata['filename']}" onclick="openImage(this.src)">
                    <div class="image-metadata">
                        <p class="filename">{metadata['filename']}</p>
                        <p>{metadata['dimensions']} | {metadata['file_size']}</p>
                        <p>Modified: {metadata['modification_time']}</p>
                    </div>
                </div>
"""
            )

        html_parts.append(
            """
            </div>
        </details>
"""
        )

    # footer
    html_parts.append(
        """
        <div class="footer">
            <p>Roman KL Pipeline</p>
        </div>
    </div>
</body>
</html>
"""
    )

    # write HTML file
    with open(output_path, 'w') as f:
        f.write(''.join(html_parts))


def generate_pdf_report(
    images_by_test: Dict[str, List[Path]],
    git_metadata: Dict[str, any],
    output_path: Path,
) -> None:
    """
    Generate multi-page PDF report using matplotlib.

    Parameters
    ----------
    images_by_test : dict
        Mapping of test_category -> list[Path].
    git_metadata : dict
        Git repository metadata from get_git_metadata().
    output_path : Path
        Path to write PDF file.

    Notes
    -----
    Creates cover page with git metadata, then one section per test.
    Images are displayed one per page with captions.
    """
    test_descriptions = get_test_descriptions()

    with PdfPages(output_path) as pdf:
        # cover page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dirty_status = 'DIRTY' if git_metadata['dirty'] else 'CLEAN'

        ax.text(
            0.5,
            0.9,
            'Roman KL Pipeline',
            ha='center',
            fontsize=24,
            weight='bold',
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.85,
            'Diagnostic Report',
            ha='center',
            fontsize=20,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.75,
            f'Generated: {timestamp}',
            ha='center',
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.65,
            'Git Repository Status',
            ha='center',
            fontsize=14,
            weight='bold',
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.60,
            f"Branch: {git_metadata['branch']}",
            ha='center',
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.56,
            f"Commit: {git_metadata['commit_hash']}",
            ha='center',
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.52,
            f"Message: {git_metadata['commit_message']}",
            ha='center',
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.48,
            f'Status: {dirty_status}',
            ha='center',
            fontsize=10,
            weight='bold',
            color='red' if git_metadata['dirty'] else 'green',
            transform=ax.transAxes,
        )

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # test sections
        for test_category in sorted(images_by_test.keys()):
            image_paths = images_by_test[test_category]
            description = test_descriptions.get(
                test_category,
                f'Diagnostic images from {test_category} test',
            )

            # section divider page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(
                0.5,
                0.6,
                test_category.replace('_', ' ').title(),
                ha='center',
                fontsize=20,
                weight='bold',
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.5,
                description,
                ha='center',
                fontsize=12,
                wrap=True,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.4,
                f'{len(image_paths)} images',
                ha='center',
                fontsize=10,
                style='italic',
                transform=ax.transAxes,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # image pages (one per image)
            for img_path in image_paths:
                metadata = get_image_metadata(img_path)

                try:
                    img_data = plt.imread(img_path)

                    fig = plt.figure(figsize=(8.5, 11))
                    ax = fig.add_subplot(111)

                    ax.imshow(img_data)
                    ax.axis('off')

                    # caption
                    caption = (
                        f"{metadata['filename']} | "
                        f"{metadata['dimensions']} | "
                        f"{metadata['file_size']}"
                    )
                    ax.text(
                        0.5,
                        -0.02,
                        caption,
                        ha='center',
                        transform=ax.transAxes,
                        fontsize=8,
                    )

                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f'Warning: Could not add {img_path} to PDF: {e}')
                    plt.close('all')

        # set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Roman KL Pipeline Diagnostics'
        d['Author'] = 'Roman KL Pipeline'
        d['Subject'] = f"Test diagnostics from branch {git_metadata['branch']}"
        d['CreationDate'] = datetime.datetime.now()


def sync_to_remote(files_to_sync: List[Path], confirm: bool = True) -> int:
    """
    Rsync files to remote server.

    Parameters
    ----------
    files_to_sync : list[Path]
        List of file paths to sync (HTML, PDF, or both).
    confirm : bool, optional
        If True, prompt user before syncing. Default is True.

    Returns
    -------
    int
        Return code (0 for success, 1 for failure).
    """
    if not files_to_sync:
        print('No files to sync')
        return 1

    # show files to be synced
    print('\nFiles to sync:')
    total_size = 0
    for f in files_to_sync:
        size_mb = f.stat().st_size / (1024**2)
        total_size += size_mb
        print(f'  - {f.name} ({size_mb:.1f} MB)')

    print(f'\nDestination: {REMOTE_HOST}:{REMOTE_DIR}/')
    print(f'Total size: {total_size:.1f} MB')

    # confirmation prompt
    if confirm:
        response = input('\nContinue with rsync? [y/N]: ')
        if response.lower() not in ['y', 'yes']:
            print('Sync cancelled')
            return 1

    # rsync each file
    print('\nSyncing files...')
    for f in files_to_sync:
        cmd = [
            'rsync',
            '-avz',
            '--progress',
            str(f),
            f'{REMOTE_HOST}:{REMOTE_DIR}/',
        ]

        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f'\nERROR: Rsync failed for {f.name}')
            print('Check SSH access to rigel server')
            return 1

    print(f'\nSuccessfully synced {len(files_to_sync)} file(s)')
    return 0


def main() -> int:
    """
    Main entry point.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()

    # default to both HTML and PDF if no format specified
    if not args.html and not args.pdf:
        args.html = True
        args.pdf = True

    # discover images
    base_dir = Path('tests/out')
    images = discover_diagnostic_images(base_dir)

    if not images:
        print('No diagnostic images found in tests/out/')
        print('Run tests first: make test-all')
        return 1

    print(
        f'Found {sum(len(v) for v in images.values())} images across {len(images)} test categories'
    )

    # get git metadata
    git_meta = get_git_metadata()

    # prepare output directory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = base_dir / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)

    # generate reports
    html_path = None
    pdf_path = None

    if args.html:
        html_path = output_dir / f'diagnostics_{timestamp}.html'
        print(f'\nGenerating HTML report: {html_path}')
        generate_html_report(images, git_meta, html_path)
        print(
            f'HTML report saved ({html_path.stat().st_size / (1024**2):.1f} MB)'
        )

        if args.open:
            print('Opening HTML report in browser...')
            webbrowser.open(f'file://{html_path.resolve()}')

    if args.pdf:
        pdf_path = output_dir / f'diagnostics_{timestamp}.pdf'
        print(f'\nGenerating PDF report: {pdf_path}')
        generate_pdf_report(images, git_meta, pdf_path)
        print(
            f'PDF report saved ({pdf_path.stat().st_size / (1024**2):.1f} MB)'
        )

    # sync to remote if requested
    if args.sync:
        # TODO: Enable if useful
        print('\nSyncing to remote server is not configured. Please set REMOTE_HOST and REMOTE_DIR in the script.')
        # files_to_sync = []
        # if html_path:
        #     files_to_sync.append(html_path)
        # if pdf_path:
        #     files_to_sync.append(pdf_path)

        # return sync_to_remote(files_to_sync, confirm=not args.no_confirm)

    return 0


if __name__ == '__main__':
    sys.exit(main())
