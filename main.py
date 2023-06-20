import re

from model import Generator, Recognizer, NaiveBayesClassifier, SemanticAnalyzer, SemanticConstructor, tokenize, chunk

input_filename = input("enter input filename: (defaults to input.txt) ")
input_filename = input_filename if input_filename != "" else "input.txt"

input_segments = []
with open(input_filename) as input_file:
    input_segments = chunk(tokenize(input_file.read()))

recognizers = {}
for label in ["nama", "kuliah", "pengalaman_intro", "pengalaman_item", "pengalaman_jobdesc", "keterampilan"]:
    recognizers[label] = Recognizer("grammars/%s.cfg" % (label))

generators = {}
for label in [{
    "name": "pengalaman_jobdesc",
    "identifier": "KalimatJobdesc"
}]:
    generators[label["name"]] = Generator("grammars/%s.cfg" % (label["name"]), label["identifier"])

classifier = NaiveBayesClassifier.load_model("data/model.json")

info = {}
info["nama"] = ""
info["kuliah"] = {"universitas": "", "fakultas": "", "jurusan": ""}
info["keterampilan"] = {"teknis": [], "sosial": []}
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
    elif state == "STMT" and label == "keterampilan":
        jenis = ""

        is_teknis = semantic.find("JenisKeterampilan").test("KeterampilanTeknis")
        is_sosial = semantic.find("JenisKeterampilan").test("KeterampilanSosial")

        if is_teknis:
            jenis = "teknis"
        elif is_sosial:
            jenis = "sosial"

        keterampilan = [k.text().title() for k in semantic.find_all("ItemKeterampilan")]
        info["keterampilan"][jenis] = keterampilan
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
            for j in semantic.find_all("KalimatJobdesc"):
                tindakan = j.find("Tindakan").text().capitalize()
                efek = j.find("KalimatEfek").text()
                tujuan = j.find("KalimatTujuan").text()

                jobdesc_sem = SemanticConstructor()
                jobdesc_sem.set("Tindakan", [tindakan])
                if efek != "":
                    jobdesc_sem.set("KalimatEfek", [efek])
                if tujuan != "":
                    jobdesc_sem.set("KalimatTujuan", [tujuan])

                jobdesc.append(generators["pengalaman_jobdesc"].generate(jobdesc_sem))

            pengalaman["jobdesc"] = jobdesc

        state = "pengalaman_item"
        info["pengalaman"][pengalaman_jenis].append(pengalaman)

def print_keterampilan(jenis, k):
    print("Keterampilan %s: %s" % (jenis, " - ".join(k)))

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

    for k in ["teknis", "sosial"]:
        print_keterampilan(k, info["keterampilan"][k])
    print()

    print("Pengalaman organisasi:")
    for p in info["pengalaman"]["organisasi"]:
        print_pengalaman(p)

    print("Pengalaman kepanitiaan:")
    for p in info["pengalaman"]["kepanitiaan"]:
        print_pengalaman(p)

print_cv()

