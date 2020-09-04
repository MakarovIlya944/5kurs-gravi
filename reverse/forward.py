import subprocess

class Forward():
  inputFile = ""

  def Build(self):
    pr = subprocess.run(["dotnet", "publish", "-c Release", "-o ../reverse/bin"])

  def PrepareInput(self):
    print("Need to copy")

  def Calculate(self):
    pr = subprocess.run(["dotnet", "run", "bin/MKE.dll"])