
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from nltk.tokenize import word_tokenize
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer



def preprocess(text):
	token_list = []
	vectorizer = TfidfVectorizer()
	factory = StemmerFactory()
	stop_factory = StopWordRemoverFactory()
	ps = factory.create_stemmer()
	stopword = stop_factory.create_stop_word_remover()
	pattern = r'[0-9]'
	text = re.sub(pattern, '', text)
	shorthand = [" no "," sy "," dg "," kk "," ga "," sdh "," bgn "," klw "," bbrp "," kbr "," dri "," dr "," jgn "," yg "," tdk ", " jt "," gk ",
				 " atw "," klu ", " tsb "," utk "," tlh "]
	replacement= [" nomor "," saya "," dengan "," kakak "," tidak "," sudah "," bangun "," kalau "," beberapa "," kabar "
		," dari "," dari "," jangan "," yang "," tidak "," juta "," tidak "," atau "," kalau "," tersebut "," untuk ", " telah"]
	for i in range(len(shorthand)):
		# re.sub(shorthand[i],replacement[i],text)
		text = text.replace(shorthand[i],replacement[i])
	text = stopword.remove(text)
	text = re.sub(r'\b\w{1,2}\b', '', text)


	processed_text = ' '.join(ps.stem(token) for token in word_tokenize(text))
	token_list.append(processed_text)
	x = vectorizer.fit_transform(token_list)

	return processed_text

#Example 
print(preprocess("SEKEDAR INFORMASI..kalau ada jeruk di jalan jgn diLindes ya, klu isinya paku ban kita bisa mledak dan nabrak, terutama di tol, atw di jalan yg sepi dan hati2 ya klu udah musibah barang2 kita di ambil….SEMOGA BERMANFAAT"))
# k,l = preprocess("GUBERNUR KALIMANTAN BARAT Pontianak, 9 November 2020 Nomor  559/1001/2-11-BKD Sifat  Penting dan SegeraLampiran  Permohonan Bantuan Dana Pengamanan Pelaksanaan Pilkada Yth. Pimpinan Direksi Perusahaan Di – Kalimantan Barat.Salam Sejahtera, Dalam rangka pelaksanaan Pemiihan Kepala Daerah (PILKADA) 2020. Pemerintah Provinsi Kalimantan Barat melalui APBD telah mengalokasikan Anggaran untuk pengamanan pelaksanaan Pilkada. Dalam NPHD yang sudah ditandatangani untuk peyelenggaraan Pilkada terdapat kekurangan dana dari nilai Anggaran yang sudah disepakati, Maka dengan ini dihimbau kepada seluruh perusahaan yang ada di Provinsi Kalimantan Barat untuk berpartisipasi dalam pembantuan dana.Hal tersebut akan diteruskan kemasing-masing Pimpinan/Direksi Perusahaan dengan memperhatikan hal-hal sebagai berikut Menginformasikan rekening donasi ke seluruh pimpinan perusahaanBank.  MANDIRI Nama Rekening  ALINAH No. Rekening  122-00-1044213-4Bantuan yang sudah terealisasi harap dilaporkan kepada Sekretaris Pemerintah Provinsi Kalimantan Barat beserta bukti pengiriman untuk diarsipkan dan diteruskan kebeberapa bidang pelaksana paling lambat 12 November 2020. Dana pengiriman dapat dikirim ke Ibu Alinah dengan nomor wa  082124223600 atau ke info.kalbarpemprov@yahoo.com Demikian disampaikan untuk dilaksanakan sebagaimana mestinya. Atas bantuan dan partisipasinya diucapkan terima kasih")
# print(l)
