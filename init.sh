#/bin/bash
chmod +rx learn_many_pytorch.sh
cd gravi/neuralNet
dotnet publish -o ~/5kurs-gravi/ --no-dependencies -c Release
# dotnet Mnist.dll learn big_1000 big_1
cd ~/5kurs-gravi