import re

from model import Generator, Recognizer, NaiveBayesClassifier, SemanticAnalyzer, SemanticConstructor, tokenize, chunk

input_filename = input("enter input filename: (defaults to input.txt) ")
input_filename = input_filename if input_filename != "" else "input.txt"

input_segments = []
with open(input_filename) as input_file:
    input_segments = chunk(tokenize(input_file.read()))

recognizers = {}
for label in ["nama", "kuliah", "pengalaman_intro", "pengalaman_item", "pengalaman_jobdesc"]:
    recognizers[label] = Recognizer("recognizers/%s.cfg" % (label))

generators = {}
for label in ["jobdesc"]:
    generators[label] = Generator("generators/%s.cfg" % (label))

classifier = NaiveBayesClassifier.load_model("data/model.json")

info = {}
info["nama"] = ""
info["kuliah"] = {"universitas": "", "fakultas": "", "jurusan": ""}
info["pengalaman"] = {"organisasi": [], "kepanitiaan": []}

# STMT, pengalaman_item, pengalaman_jobdesc
state = "STMT"
pengalaman = {}
pengalaman_jenis = ""
for segment in input_segments:
    label = classifier.classify(segment)
    syntax_tree = recognizers[label].recognize(segment)
    semantic = SemanticAnalyzer(syntax_tree)

    if state == "STMT" and label == "nama":
        info["nama"] = semantic.find("Nama").text().title()
    elif state == "STMT" and label == "kuliah":
        info["kuliah"]["universitas"] = semantic.find("FrasaUniversitas").text().title()
        info["kuliah"]["fakultas"] = semantic.find("FrasaFakultas").text().title()
        info["kuliah"]["jurusan"] = semantic.find("Jurusan").text().title()
    elif state in {"STMT", "pengalaman_item"} and label == "pengalaman_intro":
        is_organisasi = semantic.find("JenisPengalaman").test("Organisasi")
        is_kepanitiaan = semantic.find("JenisPengalaman").test("Kepanitiaan")

        if is_organisasi:
            pengalaman_jenis = "organisasi"
        elif is_kepanitiaan:
            pengalaman_jenis = "kepanitiaan"

        state = "pengalaman_item"
    elif state == "pengalaman_item" and label == "pengalaman_item":
        pengalaman = {}
        item = {}

        item["waktu_awal"] = (semantic.find("WaktuAwal").find("Bulan").get().label(), semantic.find("WaktuAwal").find("Tahun").text())
        if semantic.find("WaktuAkhir").test("Waktu"):
            item["waktu_akhir"] = (semantic.find("WaktuAkhir").find("Bulan").get().label(), semantic.find("WaktuAkhir").find("Tahun").text())
        elif semantic.find("WaktuAkhir").test("Sekarang"):
            item["waktu_akhir"] = "sekarang"
        else:
            item["waktu_akhir"] = None

        item["jabatan"] = semantic.find("Jabatan").text().title()
        item["tempat"] = semantic.find("Tempat").text().title()

        pengalaman["item"] = item

        state = "pengalaman_jobdesc"
    elif state == "pengalaman_jobdesc":
        if label == "pengalaman_jobdesc":
            jobdesc = []
            jobdesc_tree = semantic.find_all("KalimatJobdesc")
            for j in jobdesc_tree:
                tindakan = j.find("Tindakan").text().capitalize()
                efek = j.find("KalimatEfek").text()
                tujuan = j.find("KalimatTujuan").text()

                jobdesc_sem = SemanticConstructor()
                jobdesc_sem.set("VAR_Tindakan", tindakan)
                if efek != "":
                    jobdesc_sem.set("VAR_KalimatEfek", efek)
                if tujuan != "":
                    jobdesc_sem.set("VAR_KalimatTujuan", tujuan)

                jobdesc.append(generators["jobdesc"].generate(jobdesc_sem))

            pengalaman["jobdesc"] = jobdesc

        state = "pengalaman_item"
        info["pengalaman"][pengalaman_jenis].append(pengalaman)

def print_pengalaman(p):
    item = p["item"]
    jobdesc = p["jobdesc"]

    waktu = ""
    if not item["waktu_akhir"] is None:
        if type(item["waktu_akhir"]) is tuple:
            waktu = "%s %s - %s %s" % (item["waktu_awal"][0], item["waktu_awal"][1], item["waktu_akhir"][0], item["waktu_akhir"][1])
        elif item["waktu_akhir"] == "sekarang":
            waktu = "%s %s - sekarang" % (item["waktu_awal"][0], item["waktu_awal"][1])
    else:
        waktu = "%s %s" % (item["waktu_awal"][0], item["waktu_awal"][1])

    print("  %s @ %s (%s)" % (item["jabatan"], item["tempat"], waktu))

    for j in jobdesc:
        print("  - %s" % j)

    print()

def print_cv():
    print()

    print("Nama: %s" % info["nama"])
    print("Pendidikan: %s (%s, %s)" % (info["kuliah"]["universitas"], info["kuliah"]["jurusan"], info["kuliah"]["fakultas"]))
    print()

    print("Pengalaman organisasi:")
    for p in info["pengalaman"]["organisasi"]:
        print_pengalaman(p)

    print("Pengalaman kepanitiaan:")
    for p in info["pengalaman"]["kepanitiaan"]:
        print_pengalaman(p)

print_cv()

