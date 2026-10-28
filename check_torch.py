"""შეამოწმე PyTorch და პატარა გამოთვლა CPU-ზე ან Apple GPU-ზე.

გაშვება: .venv/bin/python check_torch.py
ეს არ არის მოდელის inference, სწავლება ან სიჩქარის გაზომვა.
"""

import torch

print("PyTorch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

# Tensor რიცხვების მასივია. აქ გვაქვს სამი 32-ბიტიანი ათწილადი რიცხვი.
cpu_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
cpu_result = cpu_values * 2
print("CPU result:", cpu_result.tolist())

# თუ ამ პროცესისთვის MPS ხელმისაწვდომია, იგივე გამოთვლას Apple GPU-ზე ვასრულებთ.
device = "mps" if torch.backends.mps.is_available() else "cpu"
values = cpu_values.to(device)
result = values * 2
result_on_cpu = result.cpu()

# შედეგებს CPU-ზე ვადარებთ; შეუსაბამობისას ეს ხაზი შეცდომას გამოიტანს.
torch.testing.assert_close(result_on_cpu, cpu_result)
print("Selected device:", values.device)
print("Selected device result:", result_on_cpu.tolist())
print("Result check: passed")

if device == "cpu":
    print("MPS was unavailable in this process; only CPU execution was checked.")
