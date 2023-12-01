%TRABAJO PRACTICO No1
%Carlos Fernando Quiroga Ruiz

%% PUNTO 1: 
%Dado un estado de esfuerzo definido por: 
% σ1=200 MPa, σ2=300 MPa, σ3 = -100 MPa, 
% τ12 = 50 MPa, τ13 = -80 MPa y τ23 = 100 MPa, 
% (1) Determina los esfuerzos principales. 
% (2) Determina las direcciones de los esfuerzos principales. 
% Nota: se recomienda considerar el uso de un paquete de software para manejar 
% los cálculos.

clc, clear all, close all;

o1 = 200;
o2 = 300 ;
o3 = -100;
t12 = 50;
t13 = -80; 
t23 = 100;


sigma=([[o1,t12,t13];...
        [t12,o2,t23];...
        [t13,t23,o3]])

% Calcular autovalores y autovectores los cuales nos dan las tensiones 
%principales y las direcciones de dichas tensiones
[A,autovalores, autovectores] = eig(sigma);

% Mostrar resultados
disp('Tensiones Principales:');
disp(autovalores);

disp('Direcciones principales de los esfuerzos:');
disp(autovectores);

%Otra forma de resolverlo es de la siguiente forma:
%Se calculan los tres invariantes y con ello se resuleve ecuación
%caracteristica del tensor

% Calcular los invariantes
inv1 = trace(sigma);
inv2 = 0.5 * (trace(sigma)^2 - trace(sigma^2));
inv3 = det(sigma);

% Mostrar los resultados
disp('Invariante I:');
disp(inv1);

disp('Invariante II:');
disp(inv2);

disp('Invariante III:');
disp(inv3);

%Se calculan las raices de la ecuacion caracteristica de 3er orden 
% Coeficientes de la ecuación cúbica
coeff = [1, -inv1, inv2, -inv3];

% Resolver la ecuación cúbica 
tensiones_principales = roots(coeff);

% Mostrar las raíces
disp('Las raíces de la ecuación caracteristica del tensor:');
disp(tensiones_principales);

% Se calculan las direcciones principales a partir de las tensiones
% principales calculadas
n1 = direcciones_principales(o2, o3, tensiones_principales(3), t12, t13, t23)
n2 = direcciones_principales(o2, o3, tensiones_principales(2), t12, t13, t23)
n3 = direcciones_principales(o2, o3, tensiones_principales(1), t12, t13, t23)


disp([n1, n2, n3]);
disp(autovectores);

%% PUNTO 2
clc


%% PUNTO 3
%Una capa de material compuesto unidireccional se somete a un estado de esfuerzo 
% σ1 = 245 MPa, σ2 = -175 MPa y τ12 = 95 MPa. Como se muestra en la figura 1.21, 
% las fibras en la capa de material compuesto unidireccional se orientan en un 
% ángulo θ = 25 grados con respecto al eje {1}. 
% (1) Encuentre el esfuerzo directo actuando en la dirección de la fibra. 
% (2) Encuentre el esfuerzo directo actuando en la dirección perpendicular a la fibra."

clc, clear all, close all;
imshow('1.21.jpg');
sigma_x = 245; 
sigma_y = -175;
tao12 =  95;
theta = 25;


[Sigma_Xi, Sigma_Yi, Tao_XYi] = calcular_esfuerzos(sigma_x, sigma_y, theta, tao12);

disp('El esfuerzo directo actuando en la dirección de la fibra es:');
disp(Sigma_Xi);

disp('el esfuerzo directo actuando en la dirección perpendicular a la fibra es:');
disp(Sigma_Yi);
%disp([Sigma_Xi, Sigma_Yi, Tao_XYi]);

%% PUNTO 4
% Criterio de rendimiento para una muestra cilíndrica confinada
% Considera una muestra de material homogéneo e isótropo con un coeficiente 
% de Poisson v y un límite elástico σy, confinada en un cilindro rígido, 
% como se muestra en la figura 2.2. Se aplica una única componente de 
% esfuerzo al material y se asume que no hay fricción entre la muestra y el recinto.
% (1) Encuentra el nivel de esfuerzo σ3 para el cual la muestra cederá como 
% función de σy y v si el material cumple con el criterio de Tresca. Grafica tus resultados. 
% (2) Encuentra el nivel de esfuerzo σ3 para el cual la muestra cederá como 
% función de σy y v si el material cumple con el criterio de von Mises. 
% Grafica tus resultados. Utiliza un rango de coeficientes de Poisson " en [0, 0.5].
%Ademas considerando que el anillo de confinamiento pretende hacerse de
% Acero  que espesor debería tener este anillo para no fallar antes de que
% se alcance la fluencia del bloque cilíndrico de aluminio?

clc, close all; clear all;
imshow('2.2.jpg');

% A partir de las funciones esfuerzo_maximo_tresca y esfuerzo_maximo_vonmisses
% Se calcula el esfuerzo maximo para distintos tipos de aluminio
%Aluminio 1100
% Aluminio 6061
% Aluminio 7075 Uso espacial
% Aluminio 2024

[e33T1100, sigma11T1100, Sigma33T1100] = esfuerzo_maximo_tresca(0.33 , 68000 , 55); %Aluminio 1100
[e33T6061, sigma11T6061, Sigma33T6061] = esfuerzo_maximo_tresca(0.33 , 69000 , 276); % Aluminio 6061
[e33T7075, sigma11T7075, Sigma33T7075] = esfuerzo_maximo_tresca(0.33 , 71000 , 503); % Aluminio 7075 Uso espacial
[e33T2024, sigma11T2024, Sigma33T2024] = esfuerzo_maximo_tresca(0.33 , 73000 , 324); % Aluminio 2024


[e33V1100, sigma11V1100, Sigma33V1100] = esfuerzo_maximo_vonmisses(0.33 , 68000 , 55); %Aluminio 1100
[e33V6061, sigma11V6061, Sigma33V6061] = esfuerzo_maximo_vonmisses(0.33 , 69000 , 276); % Aluminio 6061
[e33V7075, sigma11V7075, Sigma33V7075] = esfuerzo_maximo_vonmisses(0.33 , 71000 , 503); % Aluminio 7075 Uso espacial
[e33V2024, sigma11V2024, Sigma33V2024] = esfuerzo_maximo_vonmisses(0.33 , 73000 , 324); % Aluminio 2024


paso = 100;

E1min = 50000;
E1max = 80000;

V1min = 0.001;
V1max = 0.499;

SYmin = 40;
SYmax = 800;


E1 = E1min : (E1max-E1min)/(paso-1) : E1max;
V1 = V1min : (V1max-V1min)/(paso-1) : V1max;
SY = SYmin : (SYmax-SYmin)/(paso-1) : SYmax;


t1100=zeros(paso , paso);
v1100=zeros(paso , paso);


for i = 1:size(E1,2)
    for j = 1:size(V1,2)

        %Tresca
        [e33T, sigma11T, Sig33T] = esfuerzo_maximo_tresca(V1(j) , E1(i) , 503);

        t1100(i,j)=Sig33T;

        %von misses

        [e33V, sigma11V, Sig33V] = esfuerzo_maximo_vonmisses(V1(j) , E1(i) , 503); %Aluminio 1100

        v1100(i,j)=Sig33V;

    end
end

figure;
surf(V1, E1 ,t1100, 'EdgeColor', 'none');
grid on;
title('Tensión Máxima en funcion de Poisson y el Modulo Elastico (σy = 503), Tresca');
xlabel('Poisson');
ylabel('Modulo Elástico');
zlabel('Tensión Máxima');
colorbar; % Mostrar barra de colores

figure;
surf(V1, E1 ,v1100, 'EdgeColor', 'none');
grid on;
title('Tensión Máxima en funcion de Poisson y el Modulo Elastico (σy = 503), Von Misses');
xlabel('Poisson');
ylabel('Modulo Elástico');
zlabel('Tensión Máxima');
colorbar; % Mostrar barra de colores

for i = 1:size(SY,2)
    for j = 1:size(V1,2)

        %Tresca
        [e33T, sigma11T, Sig33T] = esfuerzo_maximo_tresca(V1(j) , 71000 , SY(i));

        t1100(i,j)=Sig33T;

        %von misses

        [e33V, sigma11V, Sig33V] = esfuerzo_maximo_vonmisses(V1(j) , 71000 , SY(i)); %Aluminio 1100

        v1100(i,j)=Sig33V;

    end
end

[e33V, sigma11V, Sig33V] = esfuerzo_maximo_vonmisses(0.136818 , 71000 , 62.6263)

figure;
surf( V1, SY, t1100, 'EdgeColor', 'none')
grid on;
title('Tensión Máxima en funcion de Poisson y la tension de fluencia (E = 71000), Tresca');
xlabel('Poisson');
ylabel('Tensión de Fluencia');
zlabel('Tensión Máxima');
colorbar; % Mostrar barra de colores

figure;
surf(V1, SY ,v1100, 'EdgeColor', 'none');
grid on;
title('Tensión Máxima en funcion de Poisson y la tension de fluencia (E = 71000), Von Misses');
xlabel('Poisson');
ylabel('Tensión de Fluencia');
zlabel('Tensión Máxima');
colorbar; % Mostrar barra de colores

% Punto C
% Para realizar este punto se tiene en cuenta las ecuaciones para un
% cilindro de pared gruesa al cual se le estima un radio de 30 cm y
% teniendo en cuenta la tension de fluencia del acerp A36
% Fuente: https://engineeringlibrary.org/reference/thick-pressure-vessels-air-force-stress-manual

sigmaSteel = 250; %Tension de fluencia del acero A36 (ASTM A36)
a = 0.30; %Radio Interno
b = 0.26:0.01:1.5; %Radio Externo
Pi = zeros(size(b));

for i=1:size(b,2)
    Pi(i) = sigmaSteel * (b(i)^2-a^2) / (2 * b(i)^2);
end

%wSe grafica la tension maxima interna en funcion del radio externo.
figure;
plot(b,Pi);
grid on;
title('Tensión Máxima Interna en funcion en función del radio externo');
xlabel('Radio Externo');
ylabel('Tensión Máxima Interna');


%% PUNTO 5
% considere la sencilla prueba mostrada en la figura 2.28. Se aplica un solo
% componente de esfuerzo, σ1, a una lámina con fibras dispuestas en un ángulo θ.
% La fórmula de rotación de esfuerzos (1.47) proporciona los esfuerzos aplicados 
% en la triada alineada con las fibras como 
% σ1 = σ1 cos2 θ
% σ2 = σ1 sin2 θ y 
% t12 = -σ1 cos θ sin θ. 
% El nivel de esfuerzo aplicado que corresponde al fallo satisface el
% criterio de fallo 2.97, es decir,...
% Esta ecuación de segundo orden puede resolverse para encontrar la carga de fallo. 
% Las dos soluciones corresponden a las cargas de fallo en tracción y compresión. 
% La Figura 2.29 muestra el valor absoluto de estas cargas de fallo en función 
% del ángulo de la lámina θ para los materiales Grafito/Epoxi (T300/5208) cuyas 
% propiedades se proporcionan en la tabla 2.9. Observa la caída abrupta en la 
% resistencia a medida que el ángulo de la lámina se aleja de 0 grados.

clc, close all; clear all;
% subplot(1, 2, 1);
% imshow('2.28.jpg');
% subplot(1, 2, 2);
% imshow('2.29.jpg');
% figure
% imshow('2.9.jpg');

    
theta = 0.01:0.1:90;
compresive_1 = zeros(size(theta)); %Graphite/Epoxy (T300/5208)
tensile_1 = zeros(size(theta));

compresive_2 = zeros(size(theta)); %Graphite/Epoxy (AS/3501)
tensile_2 = zeros(size(theta));

compresive_3 = zeros(size(theta)); %Boron/Epoxy (T300/5208)
tensile_3 = zeros(size(theta));

compresive_4 = zeros(size(theta)); %Scotchply (1002)
tensile_4 = zeros(size(theta));

compresive_5 = zeros(size(theta)); %Kevlar 49
tensile_5 = zeros(size(theta));


for i = 1:size(theta,2)

[compresive_1(i), tensile_1(i)] = TsaiWu_failure (1500, 1500, 40, 240, 68,theta(i)); %Graphite/Epoxy T300/5208
[compresive_2(i), tensile_2(i)] = TsaiWu_failure (1450, 1450, 52, 205, 93,theta(i)); %Graphite/Epoxy AS/3501
[compresive_3(i), tensile_3(i)] = TsaiWu_failure (1260, 2500, 61, 202, 67,theta(i)); %Boron/EpoxyT300/5208
[compresive_4(i), tensile_4(i)] = TsaiWu_failure (1060, 610,  31, 118, 72,theta(i)); %Scothply1002
[compresive_5(i), tensile_5(i)] = TsaiWu_failure (1400, 235,  12, 53,  34,theta(i)); %Kevlar 49

end

fig = figure; hold on 
plot(theta, -compresive_1,'LineWidth', 2,'DisplayName', 'Compressive strength');
plot(theta, tensile_1, 'LineWidth', 2,'DisplayName', 'Compressive strength');
ylabel('LAMINA STRENGTH');
xlabel('LAMINA ANGLE [DEGREES]');
title('Variation of the tensile and compressive failure loads with lamina angle θ., Graphite/Epoxy (T300/5208)');
legend('show');
grid on
hold off 


figure; hold on 
plot(theta, -compresive_2, 'LineWidth', 2,'DisplayName', 'Compressive strength');
plot(theta, tensile_2, 'LineWidth', 2, 'DisplayName', 'Tensile strength');
ylabel('LAMINA STRENGTH');
xlabel('LAMINA ANGLE [DEGREES]');
title('Variation of the tensile and compressive failure loads with lamina angle θ., Graphite/Epoxy AS/3501');
legend('show');
grid on
hold off

figure; hold on 
plot(theta, -compresive_3, 'LineWidth', 2,'DisplayName', 'Compressive strength');
plot(theta, tensile_3, 'LineWidth', 2,'DisplayName', 'Compressive strength');
ylabel('LAMINA STRENGTH');
xlabel('LAMINA ANGLE [DEGREES]');
title('Variation of the tensile and compressive failure loads with lamina angle θ., Boron/Epoxy (T300/5208)');
legend('show');
grid on
hold off

figure; hold on 
plot(theta, -(compresive_4), 'LineWidth', 2,'DisplayName', 'Compressive strength');
plot(theta, (tensile_4), 'LineWidth', 2,'DisplayName', 'Compressive strength');
ylabel('LAMINA STRENGTH');
xlabel('LAMINA ANGLE [DEGREES]');
title('Variation of the tensile and compressive failure loads with lamina angle θ., Scothply1002');
legend('show');
grid on
hold off

figure; hold on 
plot(theta, -(compresive_5), 'LineWidth', 2,'DisplayName', 'Compressive strength');
plot(theta, (tensile_5), 'LineWidth', 2,'DisplayName', 'Compressive strength');
ylabel('LAMINA STRENGTH');
xlabel('LAMINA ANGLE [DEGREES]');
title('Variation of the tensile and compressive failure loads with lamina angle θ., Kevlar 49');
legend('show');
grid on
hold off

figure; hold on 
plot(theta, (tensile_1),'LineWidth', 2,'DisplayName', 'Graphite/Epoxy (T300/5208)');
plot(theta, (tensile_2),'LineWidth', 2,'DisplayName', 'Graphite/Epoxy (AS/3501)');
plot(theta, (tensile_3),'LineWidth', 2,'DisplayName', 'Boron/Epoxy (T300/5208)');
plot(theta, (tensile_4),'LineWidth', 2,'DisplayName', 'Scotchply (1002)');
plot(theta, (tensile_5), 'LineWidth', 2,'DisplayName', 'Kevlar 49');
ylabel('LAMINA STRENGTH');
xlabel('LAMINA ANGLE [DEGREES]');
title('Variation of the tensile failure loads with lamina angle θ.');
legend('show');
grid on
hold off

figure; hold on 
plot(theta, -(compresive_1),'LineWidth', 2,'DisplayName', 'Graphite/Epoxy (T300/5208)');
plot(theta, -(compresive_2),'LineWidth', 2,'DisplayName', 'Graphite/Epoxy (AS/3501)');
plot(theta, -(compresive_3),'LineWidth', 2,'DisplayName', 'Boron/Epoxy (T300/5208)');
plot(theta, -(compresive_4),'LineWidth', 2,'DisplayName', 'Scotchply (1002)');
plot(theta, -(compresive_5), 'LineWidth', 2,'DisplayName', 'Kevlar 49');
ylabel('LAMINA STRENGTH');
xlabel('LAMINA ANGLE [DEGREES]');
title('Variation of the compressive failure loads with lamina angle θ.');
legend('show');
grid on
hold off

%% FUNCIONES

%Ejercicio 1 
function [nn] = direcciones_principales(oo2, oo3, oop, tt12, tt13, tt23)

    A = -(inv([[(oo2-oop),(tt23)];[(tt23),(oo3-oop)]]))*[tt12; tt13];
    nn = (1/sqrt(1 + (A(1))^2 + (A(2))^2)).*[1; A(1); A(2)];

end

%Ejercicio 3
function [Sigma_Xi, Sigma_Yi, Tao_XYi] = calcular_esfuerzos(sigma_x, sigma_y, theta, tao12)
    % Función para calcular los esfuerzos transformados
    
    theta_r = deg2rad(theta);

    % Fórmulas para calcular los esfuerzos transformados
    Sigma_Xi = (sigma_x + sigma_y) * 0.5 + (sigma_x - sigma_y) * 0.5 * cos(theta_r * 2) + tao12 * sin(theta_r * 2);
    Sigma_Yi = (sigma_x + sigma_y) * 0.5 - (sigma_x - sigma_y) * 0.5 * cos(theta_r * 2) - tao12 * sin(theta_r * 2);
    Tao_XYi = -(sigma_x - sigma_y) * 0.5 * sin(theta_r * 2) + tao12 * cos(2 * theta_r);
end

%Ejercicio 4


%function [e33, sigma11, vonmisses, vmratio, Sigma33] = esfuerzo_maximo(poisson, E, Sigma_Y)
function [e33, sigma11, Sigma33] = esfuerzo_maximo_vonmisses(poisson, E, Sigma_Y)
    % Función para realizar cálculos iterativos hasta que la relación von Mises sea igual o mayor a 1
    %E modulo elastico
    %SigmaY Tension de Fluencia
    %Sigma33 Tension en la direccion Z o 3
    %e33 Deformacion
    %sigma11 Tension en la direccion x o 1
    %vonmisses Criterio de fallo Tension equivalente<= SigmaY
    %ratio_vonmisses_SigY


    % Inicialización de variables
    Sigma33 = 1;
    vmratio = 0;

    % Bucle While: Continúa hasta que la relación von Mises sea igual o mayor a 1
    while vmratio < 1
        % Cálculo de la deformación axial (e33) usando la fórmula de deformación uniaxial
        e33 = (Sigma33 * (1 + poisson) * (1 - 2 * poisson)) / (E * (1 - poisson));

        % Cálculo del esfuerzo axial (sigma11) usando la ley de Hooke
        sigma11 = (E * poisson * e33) / ((1 + poisson) * (1 - 2 * poisson));

        % Cálculo del criterio de von Miseses
        vonmisses = sqrt((Sigma33 - sigma11)^2 + (sigma11 + Sigma33)^2);

        % Cálculo de la relación von Miseses / límite de fluencia
        vmratio = vonmisses / Sigma_Y;

        % Incremento gradual de Sigma33 en cada iteración
        Sigma33 = Sigma33 + 0.05;
    end

end

function [e33, sigma11, sigma33] = esfuerzo_maximo_tresca(poisson, E, Sigma_Y)
    % Función para realizar cálculos iterativos hasta que la relación tresca sea igual o mayor a 1
    %E modulo elastico
    %SigmaY Tension de Fluencia
    %Sigma33 Tension en la direccion Z o 3
    %e33 Deformacion
    %sigma11 Tension en la direccion x o 1
    %vonmisses Criterio de fallo Tension equivalente<= SigmaY
    %ratio_vonmisses_SigY


    % Inicialización de variables
    sigma33 = 1;
    tresca = 0;

    % Bucle While: Continúa hasta que la relación von Mises sea igual o mayor a 1
    while tresca < 1
        % Cálculo de la deformación axial (e33) usando la fórmula de deformación uniaxial
        e33 = (sigma33 * (1 + poisson) * (1 - 2 * poisson)) / (E * (1 - poisson));

        % Cálculo del esfuerzo axial (sigma11) usando la ley de Hooke
        sigma11 = (E * poisson * e33) / ((1 + poisson) * (1 - 2 * poisson));

        % Cálculo del criterio Tresca
        sum = sigma33 + sigma11;

        % Cálculo de la relación von Miseses / límite de fluencia
        tresca = sum / Sigma_Y;

        % Incremento gradual de Sigma33 en cada iteración
        sigma33 = sigma33 + 0.05;
    end

end

%Ejercicio 5

function [compresive, tensile] = TsaiWu_failure (o1t,o1c,o2t,o2c,t12,theta)

rad = deg2rad(theta);
F1 = (o1c - o1t)/sqrt(o1t*o1c);
F2 = (o2c - o2t)/sqrt(o2t*o2c);

a = ((cos(rad)^4)/(o1t*o1c)) - ((sin(rad)^2*cos(rad)^2)/sqrt(o1t*o1c*o2t*o2c)) + ((sin(rad)^4)/(o2t*o2c)) + ((sin(rad)^2*cos(rad)^2)/t12^2);
b = ((F1*cos(rad)^2)/sqrt(o1t*o1c)) + ((F2*sin(rad)^2)/sqrt(o2t*o2c));
c = -1;

p = [a b c];
r = roots(p);

compresive=r(1);
tensile = r(2);

end