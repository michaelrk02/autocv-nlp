from dataset import initialize_dataset, transform_dataset, train_dataset

dataset_source_file = "data/dataset_base.csv"
dataset_transform_file = "data/dataset_transform.csv"
model_file = "data/model.json"

print("initializing dataset ...")
initialize_dataset(dataset_source_file, [
    {
        "enable": True,
        "label": "nama",
        "semantic": {"Nama": ["X"]},
        "frequency": 1.0,
        "depth": None
    },
    {
        "enable": True,
        "label": "kuliah",
        "semantic": {"Universitas": ["X"], "Fakultas": ["x"], "Jurusan": ["X"], "Angkatan": ["X"]},
        "frequency": 0.0025,
        "depth": None
    },
    {
        "enable": True,
        "label": "pengalaman_intro",
        "semantic": {},
        "frequency": 0.25,
        "depth": None
    },
    {
        "enable": True,
        "label": "pengalaman_item",
        "semantic": {"Jabatan": ["X"], "Tempat": ["X"], "Bulan": ["X"], "Tahun": ["X"]},
        "frequency": 0.0025,
        "depth": None
    },
    {
        "enable": True,
        "label": "pengalaman_jobdesc",
        "semantic": {"Tindakan": ["X"], "KalimatEfek": ["X"], "KalimatTujuan": ["X"]},
        "frequency": 0.0025,
        "depth": 6
    },
    {
        "enable": True,
        "label": "keterampilan",
        "semantic": {"ItemKeterampilan": ["X"]},
        "frequency": 0.001,
        "depth": 6
    }
])

print("transforming dataset ...")
transform_dataset(dataset_source_file, dataset_transform_file)

print("training dataset ...")
train_dataset(dataset_transform_file, model_file)

print("all done")
