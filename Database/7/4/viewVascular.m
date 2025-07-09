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
outfile = 'PO2.gif';
path = 'C:\Users\pfran\Desktop\Vascular Response\Dataset\7\4';
cd(path)
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
% Read PO2 calculated from Vascular tumour
load(fullfile('C:\Users\pfran\Desktop\Vascular Response\Dataset\7','4P.mat'))
slice3D = P;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hFig = figure;
hAx = axes;

light('Position', [1, 1, 1]);
light('Position', [-1, -1, -1]);

hSlice = [];

% micronFactor = 4; % Conversion factor from pixels to micrometers
%
% VasosSinIrradiar = imresize3(VasosSinIrradiar, ...
%     [size(VasosSinIrradiar, 1)*micronFactor, ...
%     size(VasosSinIrradiar, 2)*micronFactor, ...
%     size(VasosSinIrradiar, 3)*micronFactor]);
%
% slice3D = imresize3(P, ...
%     [size(P, 1)*micronFactor, ...
%     size(P, 2)*micronFactor, ...
%     size(P, 3)*micronFactor]);

for zSlice = 1:size(VasosSinIrradiar,1)
    if ishandle(hSlice)
        delete(hSlice);
    end

    a = patch(isosurface(VasosSinIrradiar, 0));
    reducepatch(a, 1)
    set(a, 'facecolor', [1, 0, 0], 'edgecolor', 'none');
    set(gca, 'projection', 'perspective')
    box on
    set(gca, 'DataAspectRatio', [1, 1, 1])
    axis on
    set(gca, 'color', [1, 1, 1])

    set(gca, 'xlim', [0 size(VasosSinIrradiar, 1)], ...
        'ylim', [0 size(VasosSinIrradiar, 2)], ...
        'zlim', [0 size(VasosSinIrradiar, 3)])

    hold on

    hSlice = slice(hAx, slice3D, [], [], zSlice);
    set(hSlice,'FaceColor','interp','EdgeColor','none','DiffuseStrength',.7)
    colormap parula

    cb = colorbar('TickLabelInterpreter','Latex','FontSize',20);
    set(cb, 'Position', [0.773876511566772,0.102136752136752,0.02,0.815])
    cb.Label.String = '$PO_2$ (mmHg)'; cb.Label.FontSize = 20;
    cb.Label.Interpreter = 'latex';
    axis square

    xlabel('X axis (pixel)', 'Interpreter', 'latex')
    ylabel('Y axis (pixel)', 'Interpreter', 'latex')
    zlabel('Z axis (pixel)', 'Interpreter', 'latex')
    grid on
    grid minor
    hold off
    set(hAx,'TickLabelInterpreter','latex');
    set(hFig, 'Color', 'w')
    set(gca,'FontSize', 20)
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    view([46 20])
    xlim([0 size(P,1)])
    ylim([0 size(P,1)])
    zlim([0 size(P,1)])

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GIF creation
    set(gcf, 'color', 'w');
    drawnow;
    frame = getframe(1);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);

    if zSlice == 1
        imwrite(imind, cm, outfile, 'gif', 'DelayTime', 0, 'LoopCount', inf);
    else
        imwrite(imind, cm, outfile, 'gif', 'DelayTime', 0, 'writemode', 'append');
    end
end

close
