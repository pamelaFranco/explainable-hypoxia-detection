clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This MATLAB code visualizes a 3D dataset (P) by displaying various slices
% and a subvolume from the dataset. It starts by loading the volume data 
% from a specified path and defining the dimensions. The dataset's central
% block (subvolume) is extracted along the Z-axis and visualized as a 
% textured 3D block. The code then generates and visualizes three 
% orthogonal slices: coronal (YZ), sagittal (XZ), and axial (XY) at the 
% center of each respective axis. It replicates axial and coronal slices 
% at other positions to provide a comprehensive data view. Grid lines are
% manually plotted in 3D space for better visualization, and the color map
% is set to "parula." Labels for the axes and a color bar are added to the
% figure to indicate the PO2 values in mmHg. Finally, the code saves the 
% generated figure as a PNG image, ensuring the visualization occupies the
% entire screen.
%
%   Author:      Dr. Pamela Franco
%   Time-stamp:  2025-04-14
%   E-mail:      pamela.franco@unab.cl /pafranco@uc.cl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load oxygenation volume data
path = 'C:\Users\pfran\Desktop\Vascular Response\Dataset\7';
cd(path)
load(fullfile(pwd, '14P.mat'))

% Load vascular network volume (TIFF stack)
path = 'C:\Users\pfran\Desktop\Vascular Response\Dataset\7\14';
cd(path)
srcFiles = dir('*.tiff');
aux = imread(srcFiles(1).name);
VascularNetwork = zeros(size(aux,1), size(aux,2), length(srcFiles));
for j = 1:length(srcFiles)
    VascularNetwork(:,:,j) = imread(srcFiles(j).name);
end

% Volume dimensions
[xg, yg, zg] = size(P);
step = 10; % Grid spacing
slice_thickness = 2 * step;

% Define central Z subvolume
z_center = round(zg / 2);
z_start = max(1, z_center - step);
z_end = min(zg, z_center + step);
Psub = P(:, :, z_start:z_end);

% Create figure
hFig = figure;
hold on

% Visualize central subvolume as textured block
[X, Y, Z] = meshgrid(1:yg, 1:xg, z_start:z_end);
for k = 1:size(Psub, 3)
    surf(X(:,:,k), Y(:,:,k), Z(:,:,k), Psub(:,:,k), ...
        'EdgeColor', 'none', 'FaceColor', 'texturemap', 'FaceAlpha', 0.95);
end

% Central coronal slice (YZ plane)
x_slice = round(xg / 2);
[YY, ZZ] = meshgrid(1:yg, 1:zg);
h_coronal = surf(YY, x_slice * ones(size(YY)), ZZ, squeeze(P(x_slice, :, :))');
set(h_coronal, 'EdgeColor', 'none', 'FaceColor', 'texturemap', 'FaceAlpha', 0.95)

% Central sagittal slice (XZ plane)
y_slice = round(yg / 2);
[XX, ZZ] = meshgrid(1:xg, 1:zg);
h_sagittal = surf(y_slice * ones(size(XX)), XX, ZZ, squeeze(P(:, y_slice, :))');
set(h_sagittal, 'EdgeColor', 'none', 'FaceColor', 'texturemap', 'FaceAlpha', 0.95)

% Central axial slice (XY plane)
z_slice = round(zg / 2);
[XX, YY] = meshgrid(1:xg, 1:yg);
h_axial = surf(YY, XX, z_slice * ones(size(XX)), P(:,:,z_slice));
set(h_axial, 'EdgeColor', 'none', 'FaceColor', 'texturemap', 'FaceAlpha', 0.95)

% Grid lines for spatial reference
lineGrid = 2;
lineGridForm = 'k-';

for xi = 1:step:xg
    for yi = 1:step:yg
        plot3([xi xi], [yi yi], [1 zg], lineGridForm, 'LineWidth', lineGrid, 'Clipping', 'off');
    end
end
for yi = 1:step:yg
    for zi = 1:step:zg
        plot3([1 xg], [yi yi], [zi zi], lineGridForm, 'LineWidth', lineGrid, 'Clipping', 'off');
    end
end
for xi = 1:step:xg
    for zi = 1:step:zg
        plot3([xi xi], [1 yg], [zi zi], lineGridForm, 'LineWidth', lineGrid, 'Clipping', 'off');
    end
end

% Vascular isosurface with volume rendering
[xv, yv, zv] = meshgrid(1:size(VascularNetwork,2), 1:size(VascularNetwork,1), 1:size(VascularNetwork,3));
fv = isosurface(xv, yv, zv, VascularNetwork, 0);
isonormals(xv, yv, zv, VascularNetwork, fv.vertices);
a = patch(fv);
set(a, 'FaceColor', [1 0 0], ...      
        'EdgeColor', 'None', ...
        'FaceAlpha', 1, ...        
        'SpecularStrength', 0.4, ...
        'DiffuseStrength', 0.8, ...
        'AmbientStrength', 0.3);
lighting gouraud
material shiny
camlight('headlight')  
material([0.3 0.6 0.1 10 1])

% Axis settings
axis tight
axis equal
axis on
set(gca, 'Projection', 'perspective')
set(gca, 'DataAspectRatio', [1 1 1])
set(gca, 'Color', [1 1 1])
set(gca, 'XLim', [0 xg], 'YLim', [0 yg], 'ZLim', [0 zg])
xlabel('X axis (pixel)', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('Y axis (pixel)', 'Interpreter', 'latex', 'FontSize', 20)
zlabel('Z axis (pixel)', 'Interpreter', 'latex', 'FontSize', 20)

% Ticks and colormap
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20)
xticks([]); yticks([]); zticks([])
colormap parula
cb = colorbar('TickLabelInterpreter', 'latex', 'FontSize', 25, 'Location', 'west');
cb.Label.String = '$PO_2$ [mmHg]';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = 25;
cb.LineWidth = 2;

% Final display settings
set(hFig, 'Color', 'w')
set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1])
view([35 20])
grid on
box on
axis off

% Save the figure
cd('..')
filename = fullfile(pwd, 'PO2_Volume_Central_And_Slices.png');
saveas(gcf, filename)