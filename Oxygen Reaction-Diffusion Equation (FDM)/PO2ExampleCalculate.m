clear all, close all, clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code was developed by Dr. Pamela Franco, who, together with 
% Dr. Ignacio Espinoza as part of the master's thesis in Medical Physcis
% in the Physics Institute at Pontificia Universidad Catolica de Chile.
% We have developed a methodology for calculating the microscopic
% distribution of oxygen in tumor volumes considering realistic 3D
% vascular architectures (based on free software VascuSynth) and assessing
% the possible role of vascular damage in tumor response.
%   Author:      Dr. Pamela Franco
%   Time-stamp:  2017-01-22
%   E-mail:      pamela.franco@unab.cl /pafranco@uc.cl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path = fullfile(pwd, 'exampleImage');
cd(path)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read Vascular tumour images
capilarEtiqueta = [];
srcFiles = dir('*.tiff');
aux = imread(srcFiles(1).name);
VasosSinIrradiar = zeros(size(aux,1), size(aux,2), length(srcFiles));
for j = 1 : length(srcFiles)
    VasosSinIrradiar(:,:,j) = imread(srcFiles(j).name);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3D Oxygen distribution
P0 = VasosSinIrradiar; 
P0(P0>0) = 40;
gmax_rel = 1; 
dx = 4; 
V = length(P0)*dx; 
t_max = 30000;
tol = 1e-5;
 
tic
[P,paraGraf] = odR3(dx, V, P0, gmax_rel,t_max,tol);
toc

save(fullfile(pwd,'P02.mat'), 'P')
save(fullfile(pwd,'paraGraf.mat'), 'paraGraf')
%load(fullfile(pwd,'P02.mat'))
%load(fullfile(pwd,'paraGraf.mat'))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Iterations vs. Tolerance Condition
hFig = figure;
hAx = axes;
plot(paraGraf)
xlabel ('Iterations', 'Interpreter','latex')
ylabel('Tolerance Condition (max$\left|\!P2 - P\!\right|$)', 'Interpreter', 'latex')
set(hAx,'TickLabelInterpreter','latex');
set(gca,'FontSize', 15)
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Different Slices in PO2 volume
hFig = figure;
hAx = axes;
hAx1 = subplot(1,3,1);
imshow(P(:,:,1),[])
colormap parula
axis square
title('Slice 1', 'Fontsize', 15, 'interpreter', 'latex')
set(hAx1,'TickLabelInterpreter','latex');
hAx2 = subplot(1,3,2);
imshow(P(:,:,round(size(P,1)/2)),[])
colormap parula
axis square
title('Slice 51', 'Fontsize', 15,'Interpreter','latex')
set(hAx2,'TickLabelInterpreter','latex');
hAx3 = subplot(1,3,3);
imshow(P(:,:,size(P,1)),[])
colormap parula
title('Slice 101','Fontsize', 15,'Interpreter','latex')
set(hAx3,'TickLabelInterpreter','latex');
axis square
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
cb = colorbar('TickLabelInterpreter','Latex','FontSize',15);
set(cb, 'Position', [0.073035291798107,0.113500388500388,0.02,0.8155])
cb.Label.String = '$PO_2$ [mmHg]'; cb.Label.FontSize = 15;
cb.Label.Interpreter = 'latex';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%View 3D of PO2
hFig = figure;
hAx = axes;
[X, Y, Z] = meshgrid(1:size(P,1),1:size(P,2),1:size(P,3));
h = slice(X,Y,Z,P,floor(size(P,1)/2),floor(size(P,2)/2), ...
    floor(size(P,3)/2));
set(h,'FaceColor','interp','EdgeColor','none','DiffuseStrength',.7)
hold on
colormap gray
cb = colorbar('TickLabelInterpreter','Latex','FontSize',20);
set(cb, 'Position', [0.773876511566772,0.102136752136752,0.02,0.815])
cb.Label.String = '$PO_2$ [mmHg]'; cb.Label.FontSize = 15;
cb.Label.Interpreter = 'latex';
axis square
xlim([1 size(P,1)])
ylim([1 size(P,2)])
zlim([1 size(P,3)])
set(hAx,'TickLabelInterpreter','latex');
set(gca,'visible','off')
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca,'ztick',[])
set(hFig, 'Color', 'w')
set(gca,'FontSize', 15)
view([35 20])
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
colormap parula

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%View 3D of PO2 and Vascular Architecture
hFig = figure;
hAx = axes;
a = patch(isosurface(VasosSinIrradiar,0));
reducepatch(a,1)
set(a,'facecolor',[1,0,0],'edgecolor','none');
set(gca,'projection','perspective')
box on
light('position',[1,1,1])
light('position',[-1,-1,-1])
set(gca, 'DataAspectRatio', [1, 1, 1])
axis on
set(gca,'color',[1,1,1])
set(gca,'xlim',[0 size(VasosSinIrradiar,1)], ...
    'ylim',[0 size(VasosSinIrradiar,2)], ...
    'zlim',[0 size(VasosSinIrradiar,3)])
hold on
[X, Y, Z] = meshgrid(1:size(P,1),1:size(P,2),1:size(P,3));
h = slice(X,Y,Z,P,floor(size(P,1)/2),floor(size(P,2)/2), ...
    floor(size(P,3)/2));
set(h,'FaceColor','interp','EdgeColor','none','DiffuseStrength',.7)
cb = colorbar('TickLabelInterpreter','Latex','FontSize',20);
set(cb, 'Position', [0.773876511566772,0.102136752136752,0.02,0.815])
cb.Label.String = '$PO_2$ [mmHg]'; cb.Label.FontSize = 15;
cb.Label.Interpreter = 'latex';
axis square
colormap parula
xlim([1 size(P,1)])
ylim([1 size(P,2)])
zlim([1 size(P,3)])
set(hAx,'TickLabelInterpreter','latex');
xlabel('X [pixel]', 'Interpreter', 'latex');
ylabel('Y [pixel]', 'Interpreter', 'latex');
zlabel('Z [pixel]', 'Interpreter', 'latex');
grid on
grid minor
set(hFig, 'Color', 'w')
set(gca,'FontSize', 15)
view([35 20])
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Histogram
hFig = figure;
hAx = axes;
datos = P;
datos(datos==40) = 0;
[frecuencia, data] = hist(datos(:),16);
probabilidad = frecuencia/(length(datos)^3);
bh = bar(data,probabilidad);
bh(1).FaceColor = [32,178,170]/255;
bh(1).EdgeColor = [32,178,170]/255;
xlabel('$PO_2$ (mmHg)', 'Interpreter','latex')
ylabel('Probability', 'Interpreter','latex')
set(gca,'xtick',0:5:40)
xlim([0 40])
set(gca,'FontSize', 15)
set(hAx,'TickLabelInterpreter','latex');
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculation of vascular fraction
Vref = numel(P(P==40));
N = dx^3;
Vol = V^3;
vf = Vref*N/Vol;