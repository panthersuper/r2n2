%Autho: Itzik Ben Shabat
%Date: 10.5.15
% This scrypt illustrates the use of VoxelPlotter function to visualize
% voxel data stored in a 3d matrix

clear all
close all
clc

gridesize=256;
model = load('C:\Users\LIUChang\Desktop\train_voxels\000005\model.mat');
VoxelMat = squeeze(model.input);
disp(size(VoxelMat))
[vol_handle]=VoxelPlotter(VoxelMat,1); 

%visual effects (I recommend using the FigureRotator function from MATLAB
%Centeral
view(3);
daspect([1,1,1]);
set(gca,'xlim',[0 gridesize], 'ylim',[0 gridesize], 'zlim',[0 gridesize]);