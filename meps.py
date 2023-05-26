import requests
import xml.etree.ElementTree as ET
import csv


def get_mep_gender():
    # Find gender
    url = "https://www.tttp.eu/data/meps.csv"
    r = requests.get(url)
    if r.status_code == 200:
        csv_reader = csv.reader(r.content.decode('utf-8').splitlines(), delimiter=',')
        genders = {}
        for row in csv_reader:
            genders[row[0]] = row[6], row[2] + row[3]
        return genders


if __name__ == '__main__':
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    url = "https://www.europarl.europa.eu/meps/en/full-list/xml/"
    data = {}
    genders = get_mep_gender()
    for letter in alphabet:
        letter_url = url + letter
        r = requests.get(letter_url)
        if r.status_code != 200:
            continue
        content = r.content
        tree = ET.fromstring(content)
        if not tree:
            continue
        for mep in tree.findall('mep'):
            id = mep.find('id').text
            name = mep.find('fullName').text
            country = mep.find('country').text
            # group-ep8
            group = mep.find('politicalGroup').text
            print(id, name, country)
            try:
                gender, _ = genders.get(id, ("X", "Not found"))
                print(id, name, country, gender)
                data[id] = {"name": name, "nationality": country, "group-ep9": group, "gender": gender}
            except Exception as e:
                print("🌩️ Error " + str(e))
                continue

    outgoing = "https://www.europarl.europa.eu/meps/en/incoming-outgoing/outgoing/xml"
    r = requests.get(outgoing)
    content = r.content
    tree = ET.fromstring(content)
    for mep in tree.findall('mep'):
        id = mep.find('id').text
        name = mep.find('fullName').text
        country = mep.find('country').text
        # group-ep8
        group = mep.find('politicalGroup').text
        print("Outgoing", id, name, country)
        try:
            gender, _ = genders.get(id, ("X", "Not found"))
            print(id, name, country, gender)
            data[id] = {"name": name, "nationality": country, "group-ep9": group, "gender": gender}
        except Exception as e:
            print("🌩️ Error " + str(e))
            continue







    import json
    with open('meps.json', 'w') as f:
        json.dump(data, f, indent=2)
