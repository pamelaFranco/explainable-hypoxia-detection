clear all, close all, clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This MATLAB code visualizes and saves 3D slices and grid views of a
% dataset (P). It loads the data and then displays three slices (first,
% middle, and last) of the PO2 data with a color map. Next, it generates a 
% 3D plot of the PO2 data with grid lines overlaid, visualizing the center
% slices, and saves this as a PNG file. A second 3D plot is created to show
% the outer faces of the dataset with similar grid lines and is also saved 
% as a PNG file. The code aims to generate and save 3D visualizations of 
% the PO2 data for further analysis.
%
%   Author:      Dr. Pamela Franco
%   Time-stamp:  2025-04-14
%   E-mail:      pamela.franco@unab.cl /pafranco@uc.cl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path = 'C:\Users\pfran\Desktop\Vascular Response\Dataset\7';
cd(path)

load(fullfile(pwd,'14P.mat'))

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3D View of PO2 with Visible Grid Overlay
hFig = figure;
hAx = axes;

[X, Y, Z] = meshgrid(1:size(P,1), 1:size(P,2), 1:size(P,3));

% Slices
h = slice(X, Y, Z, P, floor(size(P,1)/2), floor(size(P,2)/2), floor(size(P,3)/2));
set(h, 'FaceColor', 'interp', 'EdgeColor', 'none', 'DiffuseStrength', .7)
%set(h, 'FaceAlpha', 0.95); 

hold on
colormap parula
cb = colorbar('TickLabelInterpreter','latex', 'FontSize', 20);
set(cb, 'Position', [0.7739, 0.1021, 0.02, 0.815])
cb.Label.String = '$PO_2$ [mmHg]';
cb.Label.FontSize = 15;
cb.Label.Interpreter = 'latex';

% Draw grid lines manually
step = 10; % Grid spacing
[xg, yg, zg] = size(P);
lineGrid = 1;
lineGridForm = 'k-';

% Z-axis grid lines
for xi = 1:step:xg
    for yi = 1:step:yg
        plot3([xi xi], [yi yi], [1 zg], lineGridForm, 'LineWidth', lineGrid, 'Clipping','off');
    end
end

% X-axis grid lines
for yi = 1:step:yg
    for zi = 1:step:zg
        plot3([1 xg], [yi yi], [zi zi], lineGridForm, 'LineWidth', lineGrid, 'Clipping','off');
    end
end

% Y-axis grid lines
for xi = 1:step:xg
    for zi = 1:step:zg
        plot3([xi xi], [1 yg], [zi zi], lineGridForm, 'LineWidth', lineGrid, 'Clipping','off');
    end
end

% Axis and appearance
axis on
axis square
view([35 20])
xlabel('X', 'Interpreter','latex', 'FontSize', 15)
ylabel('Y', 'Interpreter','latex', 'FontSize', 15)
zlabel('Z', 'Interpreter','latex', 'FontSize', 15)
xlim([1 length(X)])
ylim([1 length(Y)])
zlim([1 length(Z)])
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 15)
set(hFig, 'Color', 'w')
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

cd ..
filename = fullfile(pwd, strcat('Vascular3DGridSlicesView.png'));
saveas(gcf,filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3D View of PO2 with Visible Grid Overlay on Faces
hFig = figure;
hAx = axes;

[X, Y, Z] = meshgrid(1:size(P,1), 1:size(P,2), 1:size(P,3));

% Show outer faces instead of center slices
h = slice(X, Y, Z, P, [1 size(P,1)], [1 size(P,2)], [1 size(P,3)]);
set(h, 'FaceColor', 'interp', 'EdgeColor', 'none', 'DiffuseStrength', .7)
%set(h, 'FaceAlpha', 0.95);

hold on
colormap parula
cb = colorbar('TickLabelInterpreter','latex', 'FontSize', 20);
set(cb, 'Position', [0.7739, 0.1021, 0.02, 0.815])
cb.Label.String = '$PO_2$ [mmHg]';
cb.Label.FontSize = 15;
cb.Label.Interpreter = 'latex';

% Draw grid lines manually
step = 10; % Grid spacing
[xg, yg, zg] = size(P);
lineGrid = 1;
lineGridForm = 'k-';

% Z-axis grid lines
for xi = 1:step:xg
    for yi = 1:step:yg
        plot3([xi xi], [yi yi], [1 zg], lineGridForm, 'LineWidth', lineGrid, 'Clipping','off');
    end
end

% X-axis grid lines
for yi = 1:step:yg
    for zi = 1:step:zg
        plot3([1 xg], [yi yi], [zi zi], lineGridForm, 'LineWidth', lineGrid, 'Clipping','off');
    end
end

% Y-axis grid lines
for xi = 1:step:xg
    for zi = 1:step:zg
        plot3([xi xi], [1 yg], [zi zi], lineGridForm, 'LineWidth', lineGrid, 'Clipping','off');
    end
end

% Axis and appearance
axis on
axis square
view([35 20])
xlabel('X', 'Interpreter','latex', 'FontSize', 15)
ylabel('Y', 'Interpreter','latex', 'FontSize', 15)
zlabel('Z', 'Interpreter','latex', 'FontSize', 15)
xlim([1 xg])
ylim([1 yg])
zlim([1 zg])
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 15)
set(hFig, 'Color', 'w')
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

filename = fullfile(pwd, strcat('Vascular3DGrid.png'));
saveas(gcf,filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%