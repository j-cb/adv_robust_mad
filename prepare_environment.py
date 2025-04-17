# prepare_environment.py
import os
import subprocess
import requests
import getpass
import pyarrow.parquet as pq
from tqdm import tqdm
from huggingface_hub import login, whoami

def validate_parquet(file_path):
    """Verify if the file is a valid Parquet file"""
    try:
        pq.read_table(file_path)
        return True
    except Exception as e:
        print(f"Validation failed for {file_path}: {str(e)}")
        return False

def download_file(url, filename, max_retries=3):
    """Download a file with progress bar and retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(filename, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
            
            # Verify the downloaded file
            if validate_parquet(filename):
                return True
            else:
                print(f"Downloaded file {filename} is invalid, retrying...")
                os.remove(filename)
                
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {str(e)}")
            if os.path.exists(filename):
                os.remove(filename)
    
    return False

def install_packages():
    """Install required Python packages"""
    packages = [
        'torch',
        'transformers',
        'datasets',
        'scikit-learn',
        'seaborn',
        'matplotlib',
        'pandas',
        'tqdm',
        'requests',
        'huggingface_hub',
        'pyarrow',  # Required for Parquet validation
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call(['pip', 'install', package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

def hf_login():
    """Handle Hugging Face login with proper validation"""
    print("\n=== Hugging Face Login ===")
    print("You need a Hugging Face account with access to meta-llama/Prompt-Guard-86M")
    print("Get your token at: https://huggingface.co/settings/tokens\n")
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            token = getpass.getpass("Enter your Hugging Face token (input hidden): ")
            if not token:
                print("Token cannot be empty. Please try again.")
                continue
                
            login(token=token)
            user_info = whoami()
            print(f"\nLogged in successfully as: {user_info['name']}")
            return True
            
        except Exception as e:
            print(f"\nLogin failed: {str(e)}")
            remaining = max_attempts - attempt - 1
            if remaining > 0:
                print(f"Please try again ({remaining} attempts remaining)")
            else:
                print("Maximum attempts reached. Please check your token and try again later.")
                return False

def main():
    # Install required packages
    install_packages()
    
    # Download dataset files with validation
    base_url = "https://www.oxen.ai/synapsecai/synthetic-prompt-injections/file/main/"
    files = {
        'train': 'synthetic-prompt-injections_train.parquet',
        'test': 'synthetic-prompt-injections_test.parquet'
    }
    
    print("\nDownloading dataset files...")
    download_success = True
    for name, filename in files.items():
        if os.path.exists(filename) and validate_parquet(filename):
            print(f"{filename} already exists and is valid, skipping download")
            continue
            
        if os.path.exists(filename):
            print(f"Removing corrupted file: {filename}")
            os.remove(filename)
            
        print(f"Downloading {filename}...")
        if not download_file(base_url + filename, filename):
            print(f"Failed to download valid {filename} after multiple attempts")
            download_success = False
    
    # Handle Hugging Face login
    login_success = hf_login()
    
    print("\nSetup summary:")
    print(f"- Dataset files: {'Success' if download_success else 'Failed'}")
    print(f"- Hugging Face login: {'Success' if login_success else 'Failed'}")
    
    if download_success and login_success:
        print("\nSetup completed successfully!")
        print("You can now run the prompt_guard_AT.py training script.")
    else:
        print("\nSetup completed with some failures. You may need to:")
        if not download_success:
            print("- Manually download the dataset files from:")
            print("  https://www.oxen.ai/synapsecai/synthetic-prompt-injections")
        if not login_success:
            print("- Login to Hugging Face separately using:")
            print("  from huggingface_hub import login; login()")

if __name__ == "__main__":
    main()