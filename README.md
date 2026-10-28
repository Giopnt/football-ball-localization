# Football Ball Detection and Half-Localization

ფეხბურთის ფოტოზე ბურთის აღმოჩენა YOLO11n-ით და მისი ცენტრის მიხედვით მოედნის A/B ნახევრის განსაზღვრა. შუახაზი ხელით ინიშნება; შედეგი ნაჩვენებია Streamlit-ში და ინახება JPG/JSON ფორმატში.

[ტექნიკური ანგარიში](reports/technical_reports.md) · [სწავლებისა და შეფასების გამეორება](docs/reproduce.md) · [ექსპერიმენტების შედეგები](reports/metrics/)

## სწრაფი გაშვება — macOS

შემოწმებულია MacBook Air M4-ზე, 16 GB RAM-ით, Python 3.12.13 / PyTorch 2.11.0 / Ultralytics 8.3.220-ით. დემო იყენებს MPS-ს, ხოლო მისი მიუწვდომლობისას CPU-ს. სწავლების სკრიპტები MPS-ზეა მორგებული.

ბრძანებები გაუშვით პროექტის მთავარ საქაღალდეში. თუ `uv` არ გაქვთ:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

გახსენით ახალი ტერმინალი, შემდეგ:

```bash
uv python install 3.12.13
uv venv --python 3.12.13 --no-python-downloads .venv
uv pip sync --python .venv/bin/python requirements-macos.lock.txt
uv pip check --python .venv/bin/python
.venv/bin/python check_torch.py
```

დემოს პაკეტში მოდელი უკვე მოთავსებულია. Git-ის წყაროებიდან მუშაობისას პაკეტიდან გადმოიტანეთ checkpoint იმავე ფარდობით მისამართზე:

```text
runs/ball_yolo11n_1280_v1/weights/best.pt
```

```bash
.venv/bin/python -m streamlit run app.py \
  --server.address 127.0.0.1 \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
```

გახსენით [ადგილობრივი დემო](http://127.0.0.1:8501), აირჩიეთ ფოტო და დააჭირეთ „ბურთის მოძებნა“. გაჩერება: `Ctrl+C`. კოდის ცვლილების შემდეგ აპლიკაცია ხელახლა გაუშვით.

## ახალი ფოტო

შეინახეთ ფოტო `data/samples/` საქაღალდეში და გაუშვით:

```bash
.venv/bin/python calibrate_pitch.py --image data/samples/new_photo.jpg
```

მონიშნეთ შუახაზის ორი წერტილი და A მხარის ერთი წერტილი. `Enter` ინახავს, `R` თავიდან იწყებს, `Esc` აუქმებს. არსებულ კალიბრაციას ცვლის `--overwrite`. განაახლეთ დემოს გვერდი და აირჩიეთ ახალი ფოტო.

## მუშაობის პრინციპი

1. მოდელი აბრუნებს ბურთის კანდიდატებს: ჩარჩო, ცენტრი და confidence.
2. კანდიდატების მინიმალური ქულაა 0.05; საბოლოო არჩევანის ზღვარი — validation-ზე შერჩეული 0.332.
3. ნახევარი გამოითვლება, როცა ზღვარს ზუსტად ერთი კანდიდატი აკმაყოფილებს. სხვა შემთხვევაში შედეგია `unknown`.
4. შუახაზიდან არაუმეტეს 5 პიქსელით დაშორებული ცენტრისთვის შედეგია `near_midline`.

Inference-ის ზომაა 1280, NMS IoU — 0.7. გეომეტრია იყენებს ორიგინალი ფოტოს კოორდინატებს. A/B მხარეები კალიბრაციით განისაზღვრება.

## ძირითადი შედეგები

| შეფასება | mAP50 | mAP50–95 |
| --- | ---: | ---: |
| Pretrained, validation — 252 ფოტო | 35.79% | 12.37% |
| Fine-tuned, validation — 252 ფოტო | 73.37% | 42.30% |
| Fine-tuned, test — 91 ფოტო | 79.54% | 45.35% |

ნახევრის 12-ფოტოიან დიაგნოსტიკურ შეფასებაში (9 val / 3 test): 6 სწორი პასუხი, 6 `unknown`, 0 არასწორად დასახელებული ნახევარი. სწორი ბურთისა და სწორი ნახევრის ერთობლივი წარმატება — **6/12 (50%)**.

## ფაილები

| ფაილი | დანიშნულება |
| --- | --- |
| `app.py` | დემოს ინტერფეისი |
| `ball_pipeline.py` | inference და კანდიდატების შერჩევა |
| `half_localization.py` | ნახევრის გეომეტრიული გამოთვლა |
| `calibrate_pitch.py` | შუახაზისა და A მხარის მონიშვნა |
| `prepare_splits.py`, `train_model.py` | მონაცემების დაყოფა და fine-tuning |
| `reports/` | ანგარიში, მეტრიკები და სწავლების ჩანაწერები |

## შეზღუდვები და წყაროები

პროექტი ცალკეულ ფოტოებზე მუშაობს; შუახაზი ხელით ინიშნება. Test ნაკრები მხოლოდ ერთ კლიპს მოიცავს. Confidence კალიბრირებული ალბათობა არ არის. ჰაერში მყოფი ბურთის მოედანზე პროექცია, გუნდის ავტომატური იდენტიფიკაცია და ბურთის ფლობა არ გამოითვლება. ბურთის არმქონე კადრებზე ხარისხი ცალკე არ გაზომილა.

მონაცემები: [Roboflow football-ball-detection v1](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg/dataset/1), 903 ფოტო; გამომქვეყნებლის მითითებით CC BY 4.0. ნიმუშების ატრიბუციაა [sources.json](data/samples/sources.json)-ში, pretrained მოდელის წყარო — [model_sources.json](model_sources.json)-ში.

`demo_01` არის YouTube-ის PSG–Barcelona ვიდეოდან გადაღებული სქრინშოტი; ზუსტი ვიდეოს ბმული დაფიქსირებული არ არის. ის ცალკე სადემონსტრაციო მაგალითია და Roboflow-ის ლიცენზიის ჩანაწერი მას არ ეხება.
