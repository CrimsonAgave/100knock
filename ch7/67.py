# k-means クラスタリング
# k=5 として国名に関する単語ベクトルについてk-meansを実行する
# 国名リスト：https://datahub.io/core/country-list#data

import gensim
from sklearn.cluster import KMeans

filename = "GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)


with open("countries.csv", "r") as f:
    data = []
    for line in f:
        if(line == "Name,Code\n"): continue
        elements = line.replace("\n", "").split(",")
        data.append(elements[0])

country_vecs = []
country_names = []
for name in data:
    try:
        country_vecs.append(model[name])
        country_names.append(name)
    except:
        pass

k = 8
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(country_vecs)
for i in range(k):
    cluster = np.where(kmeans.labels_ == i)[0]
    print("cluster:", i)
    print(", ".join(country_names[j] for j in cluster))


"""

cluster: 0
Armenia, Azerbaijan, Belarus, Bulgaria, Estonia, Georgia, Kazakhstan, Kyrgyzstan, Latvia, Lithuania, Serbia, Tajikistan, Turkmenistan, Ukraine, Uzbekistan
cluster: 1
Antarctica, Australia, Bangladesh, Bhutan, Cambodia, Canada, China, India, Indonesia, Japan, Macao, Malaysia, Maldives, Mongolia, Myanmar, Nepal, Philippines, Singapore, Thailand
cluster: 2
Algeria, Angola, Benin, Botswana, Burundi, Cameroon, Comoros, Congo, Eritrea, Ethiopia, Gabon, Gambia, Ghana, Guinea, Kenya, Lesotho, Liberia, Madagascar, Malawi, Mali, Mauritania, Mozambique, Namibia, Niger, Nigeria, Rwanda, Senegal, Sudan, Swaziland, Togo, Uganda, Zambia, Zimbabwe
cluster: 3
Albania, Andorra, Austria, Belgium, Croatia, Cyprus, Denmark, Finland, France, Germany, Greece, Greenland, Hungary, Iceland, Ireland, Italy, Liechtenstein, Luxembourg, Malta, Monaco, Montenegro, Netherlands, Norway, Poland, Portugal, Romania, Slovakia, Slovenia, Spain, Sweden, Switzerland, Turkey
cluster: 4
Fiji, Guam, Kiribati, Nauru, Niue, Palau, Samoa, Tokelau, Tonga, Tuvalu, Vanuatu
cluster: 5
Anguilla, Aruba, Bahamas, Barbados, Belize, Bermuda, Curaçao, Dominica, Gibraltar, Grenada, Guadeloupe, Guernsey, Guyana, Jamaica, Jersey, Martinique, Mauritius, Mayotte, Montserrat, Pitcairn, Réunion, Seychelles, Suriname
cluster: 6
Afghanistan, Bahrain, Chad, Djibouti, Egypt, Iraq, Israel, Jordan, Kuwait, Lebanon, Libya, Morocco, Oman, Pakistan, Qatar, Somalia, Tunisia, Yemen
cluster: 7
Argentina, Brazil, Chile, Colombia, Cuba, Ecuador, Guatemala, Haiti, Honduras, Mexico, Nicaragua, Panama, Paraguay, Peru, Uruguay
"""