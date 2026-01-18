import os
import requests

URLS = {
    'train-images-idx3-ubyte.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
}

def download_file(url, dest):
    print('Downloading {}...'.format(url))
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    root = os.path.join('data', 'fashion')
    if not os.path.exists(root):
        os.makedirs(root)

    for filename, url in URLS.items():
        dest = os.path.join(root, filename)
        if not os.path.exists(dest):
            download_file(url, dest)
        else:
            print('{} already exists.'.format(filename))

    print("Done.")

if __name__ == '__main__':
    main()
