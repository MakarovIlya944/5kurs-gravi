import subprocess

class Forward():
  inputFile = ""

  def build(self):
    pr = subprocess.run(["dotnet", "publish", "-c Release", "-o ../reverse/bin"])

  def prepareInput(self):
    print("Need to copy")

  def calculate(self):
    pr = subprocess.run(["dotnet", "run", "bin/MKE.dll"])