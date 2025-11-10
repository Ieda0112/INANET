import xml.etree.ElementTree as ET
import os
import glob

# XMLフォルダパス
xml_dir = "/home/shunsuke_ieda@pip.waseda.ac.jp/workspace/2022/shunsuke_ieda/Master/COO-Comic-Onomatopoeia/COO-data/annotations"
output_dir = "COO_annos_4_INANET"  # 出力ディレクトリ
os.makedirs(output_dir, exist_ok=True)

xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))

for xml_path in xml_files:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    book_title = root.get('title')
    for page in root.find('pages').findall('page'):
        index = page.get('index')
        onos = page.findall('onomatopoeia')
        if not onos:
            continue  # オノマトペが無いページはスキップ
        annos = []
        for ono in onos:
            points = []
            i = 0
            while True:
                x = ono.get(f'x{i}')
                y = ono.get(f'y{i}')
                if x is None or y is None:
                    break
                points.append(f"{x},{y}")
                i += 1
            text = ono.text if ono.text is not None else "###"
            annos.append(','.join(points) + f",{text}")
        # indexを3桁ゼロ埋め
        index_str = str(index).zfill(3)
        out_path = os.path.join(output_dir, f"{book_title}_{index_str}.jpg")
        with open(out_path, "w", encoding="utf-8") as f:
            for line in annos:
                f.write(line + "\n")
print("変換完了: 出力先", output_dir)