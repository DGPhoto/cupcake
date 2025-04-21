#!/usr/bin/env python3
# tools/install_libraw.py

import os
import sys
import platform
import shutil
import subprocess
import tempfile
import zipfile
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("libraw_installer")

# URLs for downloading LibRaw
LIBRAW_DOWNLOADS = {
    "Windows": {
        "url": "https://www.libraw.org/data/LibRaw-0.21.1-Win64.zip",
        "extract_dir": "LibRaw-0.21.1",
        "files": ["bin/libraw.dll"]
    },
    "Darwin": {  # macOS
        "install_cmd": ["brew", "install", "libraw"],
        "check_cmd": ["brew", "list", "libraw"],
        "files": ["/usr/local/lib/libraw.dylib", "/opt/homebrew/lib/libraw.dylib"]
    },
    "Linux": {
        "debian_cmd": ["apt-get", "update", "-y", "&&", "apt-get", "install", "-y", "libraw-dev"],
        "fedora_cmd": ["yum", "install", "-y", "libraw-devel"],
        "check_cmd": ["ldconfig", "-p"],
        "files": ["/usr/lib/libraw.so", "/usr/lib64/libraw.so"]
    }
}

def is_admin():
    """Check if the script is running with administrator privileges."""
    try:
        if os.name == 'nt':  # Windows
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:  # Unix/Linux/MacOS
            return os.geteuid() == 0
    except:
        return False

def check_libraw_installed():
    """Check if LibRaw is already installed."""
    system = platform.system()
    
    if system == "Windows":
        # Check common locations
        common_paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "LibRaw", "bin", "libraw.dll"),
            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "LibRaw", "bin", "libraw.dll"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "libraw.dll"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "libraw.dll"),
        ]
        for path in common_paths:
            if os.path.exists(path):
                logger.info(f"LibRaw found at: {path}")
                return True
    
    elif system == "Darwin":  # macOS
        try:
            result = subprocess.run(LIBRAW_DOWNLOADS[system]["check_cmd"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
            if "libraw" in result.stdout.lower():
                logger.info("LibRaw is installed via Homebrew")
                return True
        except Exception as e:
            logger.debug(f"Error checking LibRaw on macOS: {e}")
            
        # Check common locations
        for path in LIBRAW_DOWNLOADS[system]["files"]:
            if os.path.exists(path):
                logger.info(f"LibRaw found at: {path}")
                return True
    
    elif system == "Linux":
        try:
            result = subprocess.run(LIBRAW_DOWNLOADS[system]["check_cmd"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
            if "libraw.so" in result.stdout.lower():
                logger.info("LibRaw is installed on the system")
                return True
        except Exception as e:
            logger.debug(f"Error checking LibRaw on Linux: {e}")
            
        # Check common locations
        for path in LIBRAW_DOWNLOADS[system]["files"]:
            if os.path.exists(path):
                logger.info(f"LibRaw found at: {path}")
                return True
    
    logger.info("LibRaw not found on the system")
    return False

def install_windows():
    """Install LibRaw on Windows."""
    logger.info("Downloading LibRaw for Windows...")
    download_data = LIBRAW_DOWNLOADS["Windows"]
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "libraw.zip")
        
        # Download the zip file
        try:
            logger.info(f"Downloading from {download_data['url']}...")
            urlretrieve(download_data["url"], zip_path)
            logger.info(f"Download completed: {os.path.getsize(zip_path)} bytes")
        except Exception as e:
            logger.error(f"Failed to download LibRaw: {e}")
            return False
        
        # Extract the zip file
        try:
            logger.info(f"Extracting zip file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            logger.info(f"Extraction completed")
        except Exception as e:
            logger.error(f"Failed to extract LibRaw: {e}")
            return False
        
        # Copy the DLL files to the target directory
        target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin")
        os.makedirs(target_dir, exist_ok=True)
        
        # Find the extracted directory
        extract_dir = os.path.join(tmp_dir, download_data["extract_dir"])
        if not os.path.exists(extract_dir):
            # Try to find the actual extraction directory
            contents = os.listdir(tmp_dir)
            dirs = [d for d in contents if os.path.isdir(os.path.join(tmp_dir, d)) and 'libraw' in d.lower()]
            if dirs:
                extract_dir = os.path.join(tmp_dir, dirs[0])
                logger.info(f"Found extraction directory: {extract_dir}")
            else:
                logger.error(f"Could not find LibRaw extraction directory. Contents: {contents}")
                return False
        
        success = False
        for file_path in download_data["files"]:
            src = os.path.join(extract_dir, file_path)
            if not os.path.exists(src):
                logger.warning(f"Source file not found: {src}")
                # Try to find the file
                for root, _, files in os.walk(extract_dir):
                    for filename in files:
                        if filename.lower() == os.path.basename(file_path).lower():
                            src = os.path.join(root, filename)
                            logger.info(f"Found file at alternate location: {src}")
                            break
                if not os.path.exists(src):
                    logger.error(f"Could not find {os.path.basename(file_path)} in extracted files")
                    continue
            
            dst = os.path.join(target_dir, os.path.basename(file_path))
            try:
                shutil.copy2(src, dst)
                logger.info(f"Copied {src} to {dst}")
                success = True
                
                # Also copy to the current directory as a fallback
                app_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
                fallback_dst = os.path.join(app_root, os.path.basename(file_path))
                if not os.path.exists(fallback_dst):
                    shutil.copy2(src, fallback_dst)
                    logger.info(f"Also copied to application root: {fallback_dst}")
            except Exception as e:
                logger.error(f"Failed to copy {src} to {dst}: {e}")
        
        if not success:
            logger.error("Failed to install any LibRaw components")
            return False
    
    logger.info("LibRaw installation completed successfully")
    return True

def install_macos():
    """Install LibRaw on macOS using Homebrew."""
    logger.info("Installing LibRaw on macOS using Homebrew...")
    
    try:
        # Check if Homebrew is installed
        subprocess.run(["brew", "--version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Homebrew is not installed. Please install Homebrew first: https://brew.sh/")
        logger.info("You can install Homebrew by running:")
        logger.info('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        return False
    
    # Install LibRaw using Homebrew
    try:
        logger.info("Running: brew install libraw")
        subprocess.run(LIBRAW_DOWNLOADS["Darwin"]["install_cmd"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=True)
        
        # Verify installation
        logger.info("Verifying installation...")
        result = subprocess.run(LIBRAW_DOWNLOADS["Darwin"]["check_cmd"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
        
        if "libraw" in result.stdout.lower():
            logger.info("LibRaw installed successfully via Homebrew")
            
            # Check if the library file exists in the expected locations
            found = False
            for lib_path in LIBRAW_DOWNLOADS["Darwin"]["files"]:
                if os.path.exists(lib_path):
                    logger.info(f"LibRaw library found at: {lib_path}")
                    found = True
            
            if not found:
                logger.warning("LibRaw installed but library file not found in expected locations")
                logger.info("Installation seems to have succeeded, but you may need to set the libraw_path in the plugin configuration")
            
            return True
        else:
            logger.error("LibRaw installation verification failed")
            return False
            
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to install LibRaw: {e}")
        
        # Try to provide more helpful error information
        if "Permission denied" in str(e):
            logger.info("Permission error encountered. You may need to run this script with sudo.")
        elif "already installed" in str(e):
            logger.info("LibRaw appears to already be installed. Checking...")
            # Check if it actually exists
            for lib_path in LIBRAW_DOWNLOADS["Darwin"]["files"]:
                if os.path.exists(lib_path):
                    logger.info(f"LibRaw library found at: {lib_path}")
                    return True
        
        return False

def install_linux():
    """Install LibRaw on Linux."""
    logger.info("Installing LibRaw on Linux...")
    
    if not is_admin():
        logger.error("Administrator privileges required. Please run with sudo.")
        logger.info("Try: sudo python tools/install_libraw.py")
        return False
    
    # Detect Linux distribution more robustly
    distro = ""
    
    # Try using lsb_release if available
    try:
        result = subprocess.run(["lsb_release", "-i"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               check=True)
        distro_info = result.stdout.lower()
        
        if "debian" in distro_info or "ubuntu" in distro_info or "mint" in distro_info:
            distro = "debian"
        elif "fedora" in distro_info:
            distro = "fedora"
        elif "centos" in distro_info or "rhel" in distro_info or "red hat" in distro_info:
            distro = "fedora"  # Use fedora-style commands for RHEL/CentOS
    except:
        # Fallback to file detection
        try:
            if os.path.exists("/etc/debian_version"):
                distro = "debian"
            elif os.path.exists("/etc/fedora-release"):
                distro = "fedora"
            elif os.path.exists("/etc/redhat-release"):
                distro = "fedora"  # Use Fedora-style commands for RHEL/CentOS
            elif os.path.exists("/etc/arch-release"):
                distro = "arch"
            elif os.path.exists("/etc/SuSE-release") or os.path.exists("/etc/opensuse-release"):
                distro = "suse"
        except:
            pass
    
    # Select appropriate installation command
    if distro == "debian":
        logger.info("Detected Debian/Ubuntu Linux distribution")
        install_cmd = ["apt-get", "update", "-y"]
        package_cmd = ["apt-get", "install", "-y", "libraw-dev"]
    elif distro == "fedora":
        logger.info("Detected Fedora/RHEL/CentOS Linux distribution")
        install_cmd = None
        package_cmd = ["yum", "install", "-y", "libraw-devel"]
    elif distro == "arch":
        logger.info("Detected Arch Linux distribution")
        install_cmd = None
        package_cmd = ["pacman", "-S", "--noconfirm", "libraw"]
    elif distro == "suse":
        logger.info("Detected openSUSE Linux distribution")
        install_cmd = None
        package_cmd = ["zypper", "install", "-y", "libraw-devel"]
    else:
        logger.error("Unsupported or undetected Linux distribution.")
        logger.info("Please install LibRaw manually using your distribution's package manager.")
        logger.info("For Debian/Ubuntu: sudo apt-get install libraw-dev")
        logger.info("For Fedora/RHEL/CentOS: sudo yum install libraw-devel")
        logger.info("For Arch Linux: sudo pacman -S libraw")
        logger.info("For openSUSE: sudo zypper install libraw-devel")
        return False
    
    # Install LibRaw
    try:
        # Run update command if needed
        if install_cmd:
            logger.info(f"Running: {' '.join(install_cmd)}")
            subprocess.run(install_cmd, check=True)
        
        # Install package
        logger.info(f"Running: {' '.join(package_cmd)}")
        subprocess.run(package_cmd, check=True)
        
        # Verify installation
        logger.info("Verifying installation...")
        verification_success = False
        
        # Check if the library files exist
        for lib_path in LIBRAW_DOWNLOADS["Linux"]["files"]:
            if os.path.exists(lib_path):
                logger.info(f"LibRaw library found at: {lib_path}")
                verification_success = True
                break
        
        # Check using ldconfig
        if not verification_success:
            try:
                result = subprocess.run(["ldconfig", "-p"], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
                if "libraw" in result.stdout.lower():
                    logger.info("LibRaw found in system libraries")
                    verification_success = True
            except:
                pass
        
        if verification_success:
            logger.info("LibRaw installed successfully")
            return True
        else:
            logger.warning("LibRaw installation might have succeeded, but verification failed")
            logger.info("You may need to manually configure the plugin if the automatic detection fails")
            return True  # Still return True as the installation command succeeded
            
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to install LibRaw: {e}")
        
        # Try to provide more helpful error information
        if "Permission denied" in str(e):
            logger.info("Permission error encountered. Make sure you're running with sudo.")
        
        return False

def verify_installation():
    """Verify the LibRaw installation by attempting to load the library."""
    logger.info("Verifying LibRaw installation...")
    
    system = platform.system()
    
    if system == "Windows":
        # Check common locations for Windows
        lib_files = ["libraw.dll", "raw.dll"]
        search_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin"),
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
            os.environ.get("PATH", "").split(os.pathsep)
        ]
    elif system == "Darwin":  # macOS
        lib_files = ["libraw.dylib"]
        search_paths = [
            "/usr/local/lib",
            "/opt/homebrew/lib",
            os.path.join(os.path.expanduser("~"), "lib"),
            os.environ.get("DYLD_LIBRARY_PATH", "").split(os.pathsep)
        ]
    else:  # Linux
        lib_files = ["libraw.so", "libraw.so.23", "libraw.so.22", "libraw.so.20", "libraw.so.19"]
        search_paths = [
            "/usr/lib",
            "/usr/local/lib",
            "/usr/lib64",
            "/usr/local/lib64",
            os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        ]
    
    # Flatten search paths
    flat_paths = []
    for path in search_paths:
        if isinstance(path, list):
            flat_paths.extend(path)
        else:
            flat_paths.append(path)
    
    # Search for library files
    for lib_file in lib_files:
        for search_path in flat_paths:
            if not search_path:
                continue
                
            full_path = os.path.join(search_path, lib_file)
            if os.path.exists(full_path):
                logger.info(f"Found LibRaw at: {full_path}")
                
                # Try to load the library
                try:
                    import ctypes
                    libraw = ctypes.cdll.LoadLibrary(full_path)
                    
                    # Check if it has the right functions
                    if hasattr(libraw, "libraw_init") and hasattr(libraw, "libraw_version"):
                        version = libraw.libraw_version()
                        if version:
                            try:
                                version_str = version.decode('utf-8')
                                logger.info(f"LibRaw version: {version_str}")
                                return True
                            except:
                                logger.info("Found LibRaw library but couldn't decode version string")
                                return True
                except Exception as e:
                    logger.debug(f"Found {full_path} but couldn't load it: {e}")
    
    logger.warning("Could not verify LibRaw installation")
    return False

def main():
    """Main function."""
    logger.info("LibRaw Installer for Cupcake")
    
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Install LibRaw for Cupcake")
    parser.add_argument("--force", action="store_true", help="Force installation even if LibRaw is already detected")
    parser.add_argument("--verify", action="store_true", help="Only verify the installation without installing")
    parser.add_argument("--yes", "-y", action="store_true", help="Automatic yes to prompts")
    args = parser.parse_args()
    
    if args.verify:
        if verify_installation():
            logger.info("LibRaw is properly installed and working")
            return 0
        else:
            logger.error("LibRaw verification failed")
            return 1
    
    if not args.force and check_libraw_installed():
        logger.info("LibRaw is already installed. No action needed.")
        logger.info("Use --force to reinstall if necessary.")
        return 0
    
    # Prompt for installation if not using --yes
    if not args.yes and not args.force and input("LibRaw not found. Install it now? (y/n): ").lower() != 'y':
        logger.info("Installation aborted by user.")
        return 0
    
    system = platform.system()
    
    if system == "Windows":
        success = install_windows()
    elif system == "Darwin":  # macOS
        success = install_macos()
    elif system == "Linux":
        success = install_linux()
    else:
        logger.error(f"Unsupported operating system: {system}")
        return 1
    
    if success:
        logger.info("LibRaw installation completed successfully")
        
        # Verify the installation
        if verify_installation():
            logger.info("Verification successful! LibRaw is properly installed.")
            logger.info("You can now use the LibRaw plugin in Cupcake")
            return 0
        else:
            logger.warning("Installation seemed successful, but verification failed")
            logger.info("You may need to configure the plugin manually with the correct library path")
            return 0  # Still return success since installation itself succeeded
    else:
        logger.error("LibRaw installation failed")
        logger.info("Please try installing LibRaw manually and configure the plugin accordingly")
        logger.info("Windows: Download from https://www.libraw.org/download")
        logger.info("macOS: brew install libraw")
        logger.info("Linux: apt-get install libraw-dev or yum install libraw-devel")
        return 1

if __name__ == "__main__":
    sys.exit(main())