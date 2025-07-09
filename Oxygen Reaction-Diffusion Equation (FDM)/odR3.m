function [P,paraGraf] = odR3(dx, V, P0, gmax_rel,t_max,tol)
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
D = 2e-9;            % m2s-1
gmax = gmax_rel*15;  % mmHg s-1
k = 2.5;             % mmHg
dt = 1e-4;           % temporal steps in 's'
n = V/dx;            % subvoxel numbers
P = P0;
ind_vessels = find(P==40);
t = 0;
aviso = 1000;
tic
paraGraf = [];
for a = t:t_max
    if (mod(a, aviso)==0)
        progreso = a/t_max;                % Progress
        transcurrido = toc;                % Time elapsed
        estimado = transcurrido/progreso;  % Estimated total time
        queda = estimado-transcurrido;     % Time remaining
        disp(sprintf('Progreso %2.2f%%. Tiempo Transcurrido: %ds. Tiempo estimado: %ds. Tiempo restante %ds.', ...
            progreso*100, round(transcurrido), round(estimado), round(queda)));
    end
    % Neumann boundary conditions
    P(1,:,:)=P(2,:,:);
    P(n,:,:)=P(n-1,:,:);
    P(:,1,:)=P(:,2,:);
    P(:,n,:)=P(:,n-1,:);
    P(:,:,1)=P(:,:,2);
    P(:,:,n)=P(:,:,n-1);
    % Laplace
    L = laplaciano3D(P,n,dx,ind_vessels);
    % Consumption
    g = consumo(P,n,gmax, k, ind_vessels);
    % Equation
    P2 = P + (D*L - g)*dt;
    ind = P2<0;
    P2(ind) = 0;
    if t > 1 && max(max(max(abs(P2 - P)))) < tol
        break;
    end
    paraGraf(length(paraGraf)+1) = max(max(max(abs(P2-P))));
    P = P2;
    t = t+1;
end
function g = consumo(P,n,gmax, k, ind_vessels)
g = gmax.*P./(P+k);
g(ind_vessels) = 0;
%Mirrow
g(1,:,:) = g(2,:,:);
g(n,:,:) = g(n-1,:,:);
g(:,1,:) = g(:,2,:);
g(:,n,:) = g(:,n-1,:);
g(:,:,1) = g(:,:,2);
g(:,:,n) = g(:,:,n-1);

function L = laplaciano3D(P,n,dx,ind_vessels)
dx = dx*1e-6; %um
L = zeros(n,n,n);
%finite difference
for i = 2:n-1
    for j = 2:n-1
        for k = 2:n-1
            L(i,j,k) = (P(i-1,j,k) + P(i+1,j,k) + P(i,j-1,k) + P(i,j+1,k)+P(i,j,k+1) + P(i,j,k-1)-6*P(i,j,k))/dx^2;
        end
    end
end
%mirrow
L(ind_vessels) = 0;
L(1,:,:) = L(2,:,:);
L(n,:,:) = L(n-1,:,:);
L(:,1,:) = L(:,2,:);
L(:,n,:) = L(:,n-1,:);
L(:,:,1) = L(:,:,2);
L(:,:,n) = L(:,:,n-1);


