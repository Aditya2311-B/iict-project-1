import urllib.request
import os
import sys

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100.0, read_so_far * 100 / total_size)
        sys.stdout.write(f"\rDownloading dataset: {percent:.2f}% completed ({read_so_far / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
    else:
        sys.stdout.write(f"\rDownloading dataset: {read_so_far / (1024*1024):.2f} MB read")
    sys.stdout.flush()

def main():
    url = "https://raw.githubusercontent.com/docketrun/Detecting-Fake-News-with-Scikit-Learn/master/fake_or_real_news.csv"
    destination = "train.csv"
    
    print(f"Starting download from: {url}")
    print(f"Saving to: {os.path.abspath(destination)}")
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=download_progress)
        print("\nDownload completed successfully!")
        
        # Verify file size and existence
        if os.path.exists(destination):
            size_mb = os.path.getsize(destination) / (1024 * 1024)
            print(f"Verified: {destination} exists, size = {size_mb:.2f} MB")
        else:
            print("Error: File was not saved successfully.")
            
    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
