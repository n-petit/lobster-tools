from arcticdb import Arctic

db_path = "/nfs/home/nicolasp/home/data/arctic"
library = "A"

arctic = Arctic(f"lmdb://{db_path}")
arctic.create_library(library)
print(arctic.list_libraries())
arctic.delete_library(library)
print(arctic.list_libraries())