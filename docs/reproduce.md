# სწავლებისა და შეფასების გამეორება

ეს ინსტრუქცია სრული რეპოზიტორიისთვისაა. გარემოს მომზადება და დემოს გაშვება აღწერილია [README-ში](../README.md). სწავლება შემოწმებულია macOS ARM64 / MPS-ზე; დამოკიდებულებების ზუსტი ვერსიებია `requirements-macos.lock.txt`-ში.

## მონაცემები

ჩამოტვირთეთ [football-ball-detection v1](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg/dataset/1) YOLOv11 ფორმატში და შეინარჩუნეთ ექსპორტის წყაროს/ლიცენზიის ფაილები. საქაღალდეები უნდა იყოს:

```text
data/raw/football_ball_v1/
├── train/images/   და train/labels/
├── valid/images/   და valid/labels/
└── test/images/    და test/labels/
```

ექსპორტი შეიცავს auto-orientation-ს; დამატებითი საექსპორტო augmentation გამოყენებული არ არის. ექსპორტის README დამატებით უთითებს [sujan-fdu5i პროექტს](https://universe.roboflow.com/sujan-fdu5i/football-ball-detection-rejhg).

```bash
.venv/bin/python prepare_splits.py
```

მოსალოდნელია train 560 / val 252 / test 91, smoke 64 / 16. დაყოფა ფიქსირებული კლიპის ID-ებით ხდება. პროექტის გადატანის შემდეგ ეს ბრძანება თავიდან გაუშვით: ის მიმდინარე ადგილმდებარეობის შესაბამის სასწავლო გზებს აგენერირებს.

## Pretrained მოდელი და fine-tuning

```bash
mkdir -p models
curl --fail --location \
  --output models/yolo11n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
.venv/bin/python train_smoke.py
.venv/bin/python train_model.py
```

Pretrained ფაილის წყარო და SHA-256 მოცემულია [model_sources.json](../model_sources.json)-ში. Fine-tuning: 20 epoch / 1280 / batch 2 / MPS. დეტალური პარამეტრებია [args.yaml](../reports/training/args.yaml)-ში.

სკრიპტი დაბეჭდავს რეალურად შექმნილ `best.pt`-ს. განმეორებით გაშვებამ შეიძლება ახალი საქაღალდე შექმნას; ასეთ შემთხვევაში განაახლეთ `WEIGHTS` იმ სკრიპტებში, რომლებითაც ახალ checkpoint-ს ამოწმებთ. Seed არ იძლევა მოწყობილობებს შორის ბიტობრივად იდენტური სწავლების გარანტიას.

## შეფასება

მოამზადეთ მონაცემები და fine-tuned checkpoint, შემდეგ:

```bash
.venv/bin/python compare_baseline.py
.venv/bin/python compare_resolutions.py
.venv/bin/python evaluate_test.py
.venv/bin/python analyze_errors.py
.venv/bin/python evaluate_half.py
```

ნახევრის შეფასება იყენებს `data/half_evaluation/`-ში შენახულ 12 მონიშვნას. მათი თავიდან მოსამზადებლად გამოიყენეთ `label_half_eval.py`. მიმდინარე შედეგების ასლებია [reports/metrics](../reports/metrics/)-ში.

## დემოს პაკეტი

```bash
.venv/bin/python package_demo.py
```

პაკეტი შეიცავს დემოს, ნიმუშებს, კალიბრაციებს, დოკუმენტაციასა და მოდელს. არსებული პაკეტის გადასაწერად განზრახ არ არის დაშვება; ახალი ვერსიისთვის შეცვალეთ `DEST` სახელი სკრიპტში.

ექსპორტისას checkpoint-ის მეტამონაცემების პირადი გზები ფარდობითი ხდება. შენახული მოდელის ტენზორები ორიგინალთან ზუსტად შედარდება; ფაილის SHA-256 მეტამონაცემების ცვლილების გამო განსხვავდება. წყაროსა და ექსპორტის ანაბეჭდები ინახება `model_manifest.json`-ში.

Git-ში არ შედის `.venv/`, `.venv-repro/`, `data/raw/`, `data/processed/`, `models/`, `runs/` და `dist/`. გასაზიარებლად გამოიყენეთ ახალი ექსპორტი; ძველი ლოკალური გაშვებები და პაკეტები შეიძლება პირად გზებს შეიცავდეს.
