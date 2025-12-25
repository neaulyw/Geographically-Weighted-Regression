import numpy as np
import matplotlib.pyplot as plt
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import netCDF4 as nc
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
import scipy.interpolate as spi
import pandas as pd
from math import sqrt

DEM_25km=nc.Dataset('G:/Data/Data/DEM_25km.nc', 'r', format = 'NETCDF4') #(9,12)
LST_25km=nc.Dataset('G:/Data/Data/LST_25km.nc', 'r', format = 'NETCDF4') #(366,9,12)
EVI_25km=nc.Dataset('G:/Data/Data/EVI_25km.nc', 'r', format = 'NETCDF4') #(366,9,12)
SM_25km=nc.Dataset('G:/Data/Data/SM_25km.nc', 'r', format = 'NETCDF4') #(366,9,12)
DEM_1km=nc.Dataset('G:/Data/Data/DEM_1km.nc', 'r', format = 'NETCDF4') #(225,300)
EVI_1km=nc.Dataset('G:/Data/Data/EVI_1km.nc', 'r', format = 'NETCDF4') #(366,225,300)
LST_1km=nc.Dataset('G:/Data/Data/LST_1km.nc', 'r', format = 'NETCDF4') #(366,225,300)
SM = pd.read_csv('G:/Data/Data/SMC.csv')

DEM_read=np.asarray(DEM_25km.variables['DEM'])
LST_read=np.asarray(LST_25km.variables['LST'])
EVI_read=np.asarray(EVI_25km.variables['EVI'])
SM_read=np.asarray(SM_25km.variables['SM'])

DEM_read_1km=np.asarray(DEM_1km.variables['DEM'])
EVI_read_1km=np.asarray(EVI_1km.variables['EVI'])
LST_read_1km=np.asarray(LST_1km.variables['LST'])


# 归一化处理
def Normalization(data):
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

LST_Nor=[]
EVI_Nor=[]
LST_1km_Nor=[]
EVI_1km_Nor=[]
for i in range(366):
    LST_Nor.append(Normalization(LST_read[i]))
    EVI_Nor.append(Normalization(EVI_read[i]))
    LST_1km_Nor.append(Normalization(LST_read_1km[i]))
    EVI_1km_Nor.append(Normalization(EVI_read_1km[i]))
LST_data=np.array(LST_Nor) # (366,9,12)
EVI_data=np.array(EVI_Nor) # (366,9,12)
LST_data_Nor=np.array(LST_1km_Nor) # (366,225,300)
EVI_data_Nor=np.array(EVI_1km_Nor) # (366,225,300)

DEM_data_Nor=Normalization(DEM_read_1km) # (225,300)
DEM_data=Normalization(DEM_read) # (9,12)

SM_data=np.array(SM_read) # (366,9,12)

y = np.asarray(SM_data).transpose(1,2,0).reshape(108,366).T
x_EVI = np.asarray(EVI_data).transpose(1,2,0).reshape(108,366).T
x_LST = np.asarray(LST_data).transpose(1,2,0).reshape(108,366).T
x_DEM = np.asarray(DEM_data).flatten()

#地理加权回归处理
coords=[]
for i in range(9):
    for j in range(12):
        coords.append([i,j])
coords_arr=np.array(coords)

Bias_Result=[]
R2_Result=[]
beta_EVI_Result=[]
beta_LST_Result=[]
beta_DEM_Result=[]
for k in range(366):
        y_arr=np.array(y[k]).reshape(108,1)
        x_Mer =np.vstack([x_EVI[k], x_LST[k],x_DEM]).T
        x_GWR = np.hstack([np.ones((108, 1)), x_Mer])
        bw = Sel_BW(coords, y_arr, x_GWR).search()
        model = GWR(coords, y_arr, x_GWR, bw)
        results = model.fit()

        Bias=np.array(results.resid_response).reshape(9,12) # 每个点的残差
        R2=np.array(results.localR2).reshape(9,12) # 每个点的局部 R²
        beta_EVI=np.array(results.params[:, 1]).reshape(9,12) # x1 回归系数
        beta_LST=np.array(results.params[:, 2] ).reshape(9,12) # x2 回归系数
        beta_DEM=np.array(results.params[:, 3]).reshape(9,12) # x2 回归系数

        Bias_Result.append(Bias)
        R2_Result.append(R2)
        beta_EVI_Result.append(beta_EVI)
        beta_LST_Result.append(beta_LST)
        beta_DEM_Result.append(beta_DEM)

Bias_Result=np.array(Bias_Result)
R2_Result=np.array(R2_Result)
beta_EVI_Result=np.array(beta_EVI_Result)
beta_LST_Result=np.array(beta_LST_Result)
beta_DEM_Result=np.array(beta_DEM_Result)

# print(Bias_Result.shape,R2_Result.shape,beta_EVI_Result.shape,beta_LST_Result.shape,beta_DEM_Result.shape)
# plt.figure(figsize=(7,5))
# sns.heatmap(R2_Result,annot=False,cmap="cool",linewidths=0.3,linecolor="grey")
# plt.title('R2', fontsize=14)
# plt.savefig('G:/Data/Data/R2.jpg')
# plt.show()

# 三次样条插值 cubic
def Interpolation(Input_data, scal):
    n, m = Input_data.shape
    iy_a = []
    for i in range(n):
        x = np.linspace(start=1, stop=m, num=m)
        x = np.array(x)
        y = np.array(Input_data[i])
        ix = np.linspace(start=1, stop=m, num=m * scal)
        ix = np.array(ix)
        ipo = spi.splrep(x, y, k=3)
        iy = spi.splev(ix, ipo)
        iy_a.append(iy)
    result = np.array(iy_a)
    return result
#
scal=25 #分辨率扩大25倍

beta_DEM_1km=[]
beta_EVI_1km=[]
beta_LST_1km=[]
Bias_1km=[]
for x in range(366):
    Inter_DEM=np.asarray(Interpolation(beta_DEM_Result[x],scal)).T #(225,300)
    beta_DEM_1km_r=np.asarray(Interpolation(Inter_DEM,scal)).T
    beta_DEM_1km.append(beta_DEM_1km_r)

    Inter_EVI=np.asarray(Interpolation(beta_EVI_Result[x],scal)).T #(225,300)
    beta_EVI_1km_r=np.asarray(Interpolation(Inter_EVI,scal)).T
    beta_EVI_1km.append(beta_EVI_1km_r)

    Inter_LST=np.asarray(Interpolation(beta_LST_Result[x],scal)).T #(225,300)
    beta_LST_1km_r=np.asarray(Interpolation(Inter_LST,scal)).T
    beta_LST_1km.append(beta_LST_1km_r)

    Inter_Bias=np.asarray(Interpolation(Bias_Result[x],scal)).T #(225,300)
    Bias_1km_r=np.asarray(Interpolation(Inter_Bias,scal)).T
    Bias_1km.append(Bias_1km_r)

beta_DEM_1km=np.array(beta_DEM_1km)
beta_EVI_1km=np.array(beta_EVI_1km)
beta_LST_1km=np.array(beta_LST_1km)
Bias_1km=np.array(Bias_1km)

# print(beta_DEM_1km.shape,beta_EVI_1km.shape,beta_LST_1km.shape,Bias_1km.shape)
#
DN=[]
for a in range(366):
    for b in range(225):
        for c in range(300):
            DN_value=DEM_data_Nor[b,c]*beta_DEM_1km[a,b,c]+LST_data_Nor[a,b,c]*beta_LST_1km[a,b,c]+EVI_data_Nor[a,b,c]*beta_EVI_1km[a,b,c]+Bias_1km[a,b,c]
            DN.append(DN_value)
SM_1km=np.array(DN).reshape(366,225,300) #1km逐日土壤水分最终结果

# #空白值替换为nan
# coords_0=[[0,0],[0,1],[0,3],[0,4],[0,5],[0,8],[0,9],[0,10],[0,11],
#           [1,10],[1,11],
#           [2,11],
#           [3,0],[3,11],
#           [4,0],
#           [5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],
#           [6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6],
#           [7,0],[7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7],[7,8],
#           [8,0],[8,1],[8,2],[8,3],[8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[8,10]]
# coords_0_Arr=np.array(coords_0)
#
# for p in range(366):
#     for coord in coords_0_Arr:
#         m,n = coord
#         SM_1km[p,m, n] = 0
# SM_1km=np.where(SM_1km,SM_1km,np.nan)

#
# # #晴空像元比例
# # def Percent(file_names,Nodata):
# #     per = []
# #     for m in range(366):
# #             data = np.array(file_names)
# #             sum_count = np.count_nonzero(data[m] > 0)
# #             P = sum_count / Nodata
# #             per.append(P)
# #     return per
# #
# # a=np.count_nonzero(SM_1km[1] > 0)
# # SM_1km_P = np.array(Percent(SM_1km,Nodata=a))
# #
# # xbar=np.arange(1,367,1)
# # ybar=SM_1km_P
# # plt.figure(figsize=(20, 6))
# # plt.scatter(xbar,ybar)
# # plt.xlabel('DOY',fontsize=14)
# # plt.ylabel('Percent',fontsize=14)
# # ax=plt.gca()
# # ax.yaxis.set_major_locator(MultipleLocator(0.2))
# # plt.ylim(-0.2,1.2)
# # plt.savefig('C:/Users/Yuwan/Desktop/3/Percent_GWR.jpg')
#
# 保存最终的SM
input_NCdata=np.array(SM_1km) #输入数据
lon=np.asarray(DEM_1km.variables['lon']) # refer
lat=np.asarray(DEM_1km.variables['lat']) # refer
ncfile = nc.Dataset('G:/Data/Data/SM_1km.nc' ,'w' ,format = 'NETCDF4') #保存NC路径

# 添加坐标轴（经度纬度和时间）
xdim = ncfile.createDimension('lon' ,300) #300
ydim = ncfile.createDimension('lat' ,225) #225
tdim = ncfile.createDimension('time',366)

# # 添加全局属性，比如经纬度和标题，主要是对数据进行一个简单的介绍
ncfile.setncattr_string('title' ,'TEMPERATURE')
ncfile.setncattr_string('geospatial_lat_min' ,'-36.476 degrees')
ncfile.setncattr_string('geospatial_lat_max' ,'-34.476 degrees')
ncfile.setncattr_string('geospatial_lon_min' ,'146.655 degrees')
ncfile.setncattr_string('geospatial_lon_max' ,'149.405 degrees')
#
# # 添加变量和局部属性，存入数据
var = ncfile.createVariable(varname='lon' ,datatype=np.float64,dimensions='lon')
var.setncattr_string('long_name' ,'longitude')
var.setncattr_string('units' ,'degrees_east')
var[: ] =lon
#
var = ncfile.createVariable(varname='lat' ,datatype=np.float64 ,dimensions='lat')
var.setncattr_string('long_name' ,'latitude')
var.setncattr_string('units' ,'degrees_north')
var[: ] =lat
#
tvar = ncfile.createVariable(varname='time', datatype=np.float64 ,dimensions='time')
tvar.setncattr_string('long_name' ,'time')
tvar.setncattr_string('units' ,'days since 0000-01-01')
tvar.calendar = "standard"
tvar[: ] =366
#
var = ncfile.createVariable(varname='SM' ,datatype=np.float64 ,dimensions=('time' ,'lat' ,'lon'))
var.setncattr_string('long_name' ,'SM')
var.setncattr_string('units' ,'%')
var[: ] =input_NCdata

# 关闭文件
ncfile.close()
print('finished')