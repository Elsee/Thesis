data = csvread("featuresOrig_Walking2.csv", 1);
data1 = transp(horzcat(data(:,1:12), data(:,19:27)));
data2 = transp(data(:,28:36));
data3 = transp(data(:,13:18));
data4 = transp(data(:,37:57));

data1 = myNorm(data1, 1);
data2 = myNorm(data2, 1);
data3 = myNorm(data3, 1);
data4 = myNorm(data4, 1);

[AE, Error, EncoderOutput] = LBPGFAutoEncoder(data1, data2, data3, data4, 3000, 10, 4, 3, 10, 15);

csvwrite('Eoutput2.csv', EncoderOutput);