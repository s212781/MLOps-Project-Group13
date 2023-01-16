from google.cloud import storage
import dvc.api
import dill
import timm
from timm.models import create_model
from timm.data.loader import create_loader
from dvc.api import DVCFileSystem
print(dvc.api.get_url('data/'))
# bucket_name = 'dtu-mlops-m21-1'
# prefix = 'your-bucket-directory/'
# dl_dir = 'your-local-directory/'

# storage_client = storage.Client()
# bucket = storage_client.get_bucket(bucket_name=bucket_name)
# blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
# for blob in blobs:
#     filename = blob.name.replace('/', '_') 
#     blob.download_to_filename(dl_dir + filename)  # Download

# data_set = dvc.api.get_url('data/proccessed/')
# train_loader = create_loader(dataset, (3, 224, 224), batch_size=32, is_training=True, use_prefetcher=False)
# test_loader = create_loader(dataset, (3, 224, 224), batch_size=32, is_training=False, use_prefetcher=False)
url = dvc.api.get_url('data/')
fs = DVCFileSystem(url, rev="main")
fs.find("/", detail=False, dvc_only=True)