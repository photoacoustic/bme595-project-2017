import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
import math

import numpy as np

from torch.utils.data import TensorDataset, DataLoader
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(91,182,kernel_size=4, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),


            nn.Conv2d(182, 364, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier= nn.Sequential(

            nn.Linear(364*5*6,4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000, bias=False),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.result = nn.Linear(1000, 1)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 10920)
        x = self.classifier(x)
        x = self.result(x)
        return x


model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
Nocoffee=['13','16','19','22','25','26','28','31','34','37','39','42','45','47','50','54','55','56','58','60','62','64','65','67','68','69','70','72','74','76','78','79','81','83','85','86','88','91','93','94','95','97','99','101','104'];
Coffee=['14','15','17','18','20','21','23','24','27','29','30','32','33','35','36','38','40','41','43','44','46','48','49','51','53','57','59','61','63','66','71','73','75','77','80','82','84','87','89','92','96','98','100','102','103'];
parcellation = ['12','14','23','36','53','84','96','97','102','106','108','113','117','120','121','132','153','156','165','168','184','185','190','205','208','214','217','222','223','227','235','236','247','257','270','4','7','15','18','28','32','34','44','45','50','55','56','71','83','93','105','139','148','154','157','183','210','218','230','246','253','254','255','262','264','268','277','278','8','29','35','40','41','57','60','64','80','85','87','100','146','147','160','166','206','229','232','240','242','243','250','275','1','5','17','27','59','82','91','99','137','169','172','173','195','207','216','224','237','245','269','3','22','70','74','109','112','123','163','170','181','188','213','265','274','16','20','67','72','88','90','94','107','116','124','125','129','134','138','140','143','145','152','176','177','179','180','187','193','194','197','201','203','209','215','221','226','249','259','276','2','6','9','10','19','21','24','26','33','37','39','43','46','47','48','51','54','62','63','65','69','75','78','81','89','92','95','101','103','111','119','122','131','135','161','164','171','182','186','189','192','198','200','204','212','225','228','234','238','241','244','248','251','258','267','271','11','25','30','42','49','52','58','77','98','114','115','118','127','128','133','136','144','149','151','155','158','167','191','199','233','239','260','261','263','266','272','273','13','31','38','61','66','68','73','76','79','86','104','110','126','130','141','142','150','159','162','174','175','178','196','202','211','219','220','231','252','256']

TraindataS=1
TraindataE=80
CNs=0 #CNs+1 = real started parcellation
CNe=279 #CNe-1 = real end parcellation
EpochN=100
TestdataS=95
TestdataE=105
testCNs=CNs
testCNe=CNe
# no train
def train():
    validaterate = []
    trainrate = []
    for epoch in range(EpochN):
        #Train
        point = 0
        count = 0
        for x_batch, y_batch in loader:

            x_var = Variable(x_batch)
            y_var = Variable(y_batch)
            y_pred = model(x_var)
            optimizer.zero_grad()
            loss = criterion(y_pred, y_var)
            loss.backward()
            optimizer.step()
            if math.isnan(y_pred.data[0][0]):
                y_pred.data[0][0]=0
                #print("Hi I am nan, I am leaving, now y_pred.data = ", y_pred.data[0][0])
            else:
                if y_var.data[0][0] == 1:
                    point += torch.abs(y_pred.data)
                    count += 1
                elif y_var.data[0][0] == 0:
                    point += (1 - torch.abs(y_pred.data))
                    count += 1
                else:
                    print("no", i, "days found")
                    continue;
        if count == 0:
            count =1
            print("Training rate for epoch " + str(epoch + 1) + " = ", point / count)
            trainrate.append(point / count)
        else:
            print("Training rate for epoch " + str(epoch+1) + " = ", point[0][0]/count)
            trainrate.append(point[0][0]/count)
        torch.save(model,'mytraining5.pt')

        #Test
        test_point = 0
        test_count = 0
        for test_x_batch, test_y_batch in test_loader:
            test_x_var = Variable(test_x_batch)
            test_y_var = Variable(test_y_batch)
            test_y_pred = model(test_x_var)
            if math.isnan(test_y_pred.data[0][0]):
                test_y_pred.data[0][0]=0;
            else:
                if test_y_var.data[0][0] == 1:
                    test_point += torch.abs(test_y_pred.data)
                    test_count += 1
                elif test_y_var.data[0][0] == 0:
                    test_point += (1 - torch.abs(test_y_pred.data))
                    test_count += 1
                else:
                    print("no", i, "days found")
                    continue;
        if test_count == 0:
            test_count =1
            print("Validation rate for epoch " + str(epoch + 1) + " = ", test_point / test_count)
            validaterate.append(test_point / test_count)
        else:
            print("Validation rate for epoch "+str(epoch+1)+" = ", test_point[0][0]/test_count)
            validaterate.append(test_point[0][0]/test_count)
        print("end of epoch", epoch + 1)
        print("-------------------------------")
    plt.plot(trainrate, label="training line")
    plt.plot(validaterate, label="validating line")
    plt.ylabel('Accuracy_rate')
    plt.legend(loc='upper left')
    plt.xlabel('epoch')
    plt.show()

# Train data
fileObj = open("/Users/yang1399/Documents/dDb/myconn/rEg_directly_filter_func/parcellasion_timeseries_spaceVer/13/parcellation_"+ parcellation[CNs]+".txt")
y = np.loadtxt(fileObj)
y = torch.from_numpy(y).float()
Y = y.view([1, 91, 109, 91])
Z= torch.zeros(1, 1)
for i in range(TraindataS,TraindataE):
    if str(i) in Coffee:
        #Z = torch.ones(278,1)
        z = torch.ones(CNe-CNs, 1)
    elif str(i) in Nocoffee:
        #Z = torch.zeros(278,1)*100
        z = torch.zeros(CNe-CNs, 1)
    else:
        print("no",i,"days found")
        continue;
    Z= torch.cat((Z,z),0)
    for j in range(CNs,CNe):
        fileObj = open("/Users/yang1399/Documents/dDb/myconn/rEg_directly_filter_func/parcellasion_timeseries_spaceVer/"+str(i)+"/parcellation_"+ parcellation[j + 1]+".txt")
        y = np.loadtxt(fileObj)
        y = torch.from_numpy(y).float()
        y = y.view([1, 91, 109, 91])
        fileObj.close()
        Y = torch.cat((Y, y), 0)
    print("Done loading training day ", i)
loader = DataLoader(TensorDataset(Y, Z), batch_size=1) #Y data, Z label
print("-------------------------------")

# Test data
fileObj_test = open("/Users/yang1399/Documents/dDb/myconn/rEg_directly_filter_func/parcellasion_timeseries_spaceVer/13/parcellation_"+ parcellation[CNs]+".txt")
test_y = np.loadtxt(fileObj_test)
test_y = torch.from_numpy(test_y).float()
test_Y = test_y.view([1, 91, 109, 91])
test_Z = torch.zeros(1, 1)
for k in range(TestdataS, TestdataE):
    if str(k) in Coffee:
        test_z = torch.ones(testCNe - testCNs, 1)
    elif str(k) in Nocoffee:
        test_z = torch.zeros(testCNe - testCNs, 1)
    else:
        print("no", k, "days found")
        continue;
    test_Z = torch.cat((test_Z, test_z), 0)
    for j in range(testCNs, testCNe):
        fileObj_test = open("/Users/yang1399/Documents/dDb/myconn/rEg_directly_filter_func/parcellasion_timeseries_spaceVer/" + str(k) + "/parcellation_"+parcellation[j + 1]+".txt")
        test_y = np.loadtxt(fileObj_test)
        test_y = torch.from_numpy(test_y).float()
        test_y = test_y.view([1, 91, 109, 91])
        fileObj_test.close()
        test_Y = torch.cat((test_Y, test_y), 0)
    print("Done loading testing day ", k)
test_loader = DataLoader(TensorDataset(test_Y, test_Z), batch_size=1)  # Y data, Z label
print("=============================")
print("prepared to train")

#Let's train
train()

'''
# It works!
# Show testing output of 277
DDay=15
fileObj = open("/Users/yang1399/Documents/dDb/myconn/rEg_directly_filter_func/parcellasion_timeseries_spaceVer/"+str(DDay)+"/parcellation_1.txt")
y = np.loadtxt(fileObj)
y = torch.from_numpy(y).float()
Y = y.view([1,91,109,91])
for i in range(277):

    fileObj = open("/Users/yang1399/Documents/dDb/myconn/rEg_directly_filter_func/parcellasion_timeseries_spaceVer/"+str(DDay)+"/parcellation_",parcellation[j + 2],".txt")
    y = np.loadtxt(fileObj)
    y = torch.from_numpy(y).float()
    y = y.view([1,91,109,91])
    fileObj.close()
    Y = torch.cat((Y,y),0)

Z = torch.zeros(278,1)
loader = DataLoader(TensorDataset(Y, Z), batch_size=1)
model = Net()
model = torch.load("mytraining5.pt")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
print(model(Variable(Y)))
'''