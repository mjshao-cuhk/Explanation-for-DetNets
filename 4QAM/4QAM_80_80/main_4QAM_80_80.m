clear; close all; clc;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program performs Monte-carlo simulation on the Bit Error Rate(BER)
% performance of MIMO detectors in the paper 
% "An Explanation of Deep MIMO Detection from a Perspective of Homotopy
% Optimization" by Mingjie Shao, Wing-Kin Ma and Junbin Liu.
% settings: 4QAM,  80 by 80 MIMO (real-sized)
% last updated on Jan. 30, 2023
% If you have any questions, please contact {mjshao,junbinliu}@link.cuhk.edu.hk

%%
%rng(0);
M = 40; N = 40; % N - number transmit symbols, M - receive dimension.
QAM_size = 4;     %QAM size
mod_ord = log2(QAM_size);
SNR = 0:2:20;   %SNR in dB
max_no_simulations = 1000; % No. of Monte-carlo simulations used

%-------------------------
sigma_snr = sqrt( N*10 .^ ( - (SNR) / 10 ) ); % MMSE coefficient.
u=sqrt(QAM_size)-1;         %symbol bound 
Es=2/3*(QAM_size-1);        %average energy of a QAM symbol
cons = (1:u+1)*2-u-2;       %symbol vector


%variables for storing the SERs
ber_NI = zeros(length(SNR),1);
ber_Box = zeros(length(SNR),1);
ber_PGunfold = zeros(length(SNR),1);
ber_PG = zeros(length(SNR),1);
ber_FWunfold = zeros(length(SNR),1);
ber_FW = zeros(length(SNR),1);
ber_ADMMunfold = zeros(length(SNR),1);
ber_ADMM = zeros(length(SNR),1);


%time
t_NI = zeros(length(SNR),1);
t_Box = zeros(length(SNR),1);
t_PGunfold = zeros(length(SNR),1);
t_PG = zeros(length(SNR),1);
t_FWunfold = zeros(length(SNR),1);
t_FW = zeros(length(SNR),1);
t_ADMMunfold = zeros(length(SNR),1);
t_ADMM = zeros(length(SNR),1);


%% load parameters
 
load('.\PG_unfold_par\parameter0.mat')
load('.\PG_unfold_par\parameter1.mat')
load('.\PG_unfold_par\parameter2.mat')
PGpar0 = parameter0; PGpar1 = parameter1; PGpar2 = parameter2;

load('.\FW_unfold_par\parameter0.mat')
load('.\FW_unfold_par\parameter1.mat')
load('.\FW_unfold_par\parameter2.mat')
FWpar0 = parameter0; FWpar1 = parameter1; FWpar2 = parameter2;

load('.\ADMM_unfold_par\parameter0.mat')
load('.\ADMM_unfold_par\parameter1.mat')
load('.\ADMM_unfold_par\parameter2.mat')
ADMMpar0 = parameter0; ADMMpar1 = parameter1; ADMMpar2 = parameter2;
 
for snr_pt = 1 : (length(SNR)) 
    for simulation_no = 1 : max_no_simulations
       
        %-----------set up channels, tx & rx signals , etc-------------
        % Generate QAM symbols
        s=randi(u+1,2*N,1);
        s=2*s-u-2;
        s_c= s(1:N)+sqrt(-1)*s(N+1:end);
        s_bit = qamdemod(s_c, 2^mod_ord,'OutputType','bit');
        % generate the random channel
        H_c = (1/sqrt(2))* (randn(M,N) + 1i*randn(M,N));
        % generate the output vector
        n_c = (sigma_snr(snr_pt)*sqrt(Es)/sqrt(2))*(randn(M,1)+1i*randn(M,1));
        y_c = H_c*s_c + n_c;
        % Complex to real conversion
        y = [real(y_c); imag(y_c)]; H = [real(H_c) -imag(H_c); imag(H_c) real(H_c)];
        
        %--------detectors------------------

        x_ini=pinv(H)*y; 
        HH = H.'*H;
        Hy = H.'*y;
        Lf = 2*norm(HH);

        % no interference;
        tic
        sc_NI= NoInterference(H_c,s_c,n_c);
        t_NI(snr_pt) = t_NI(snr_pt) + toc;
        temp = [real(sc_NI); imag(sc_NI)];
        s_NI = deQAM(temp, cons);
        NI_bit = qamdemod(sc_NI, 2^mod_ord,'OutputType','bit');
        ber_NI(snr_pt) = ber_NI(snr_pt)+sum(NI_bit~=s_bit);

        % Box;
        tic
        s_Box = Box_rel(x_ini, HH, Hy, Lf, cons);
        t_Box(snr_pt) = t_Box(snr_pt) + toc;  
        x_ini_box = s_Box;
        s_Box = deQAM(s_Box, cons);
        sc_Box = s_Box(1:N)+sqrt(-1)*s_Box(N+1:2*N);
        Box_bit = qamdemod(sc_Box, 2^mod_ord,'OutputType','bit');
        ber_Box(snr_pt) = ber_Box(snr_pt)+sum(Box_bit~=s_bit);  

        % HoT-PG;
        tic
        s_PG = HoT_PG(x_ini, HH, Hy, Lf);
        t_PG(snr_pt) = t_PG(snr_pt) + toc;
        s_PG = deQAM(s_PG, cons);
        sc_PG = s_PG(1:N)+sqrt(-1)*s_PG(N+1:2*N);
        PG_bit = qamdemod(sc_PG, 2^mod_ord,'OutputType','bit');
        ber_PG(snr_pt) = ber_PG(snr_pt)+sum(PG_bit~=s_bit); 

        % PG DetNet;
        tic
        s_PGunfold = PG_DetNet(x_ini, HH, Hy, 20, PGpar0, PGpar1, PGpar2);
        t_PGunfold(snr_pt) = t_PGunfold(snr_pt) + toc;
        s_PGunfold = deQAM(s_PGunfold, cons);
        sc_PGunfold = s_PGunfold(1:N)+sqrt(-1)*s_PGunfold(N+1:2*N);
        PGunfold_bit = qamdemod(sc_PGunfold, 2^mod_ord,'OutputType','bit');
        ber_PGunfold(snr_pt) = ber_PGunfold(snr_pt)+sum(PGunfold_bit~=s_bit); 

        % HoT-FW;
        tic
        s_FW = HoT_FW(x_ini_box, HH, Hy, H, y, Lf);
        t_FW(snr_pt) = t_FW(snr_pt) + toc;
        s_FW = deQAM(s_FW, cons);
        sc_FW = s_FW(1:N)+sqrt(-1)*s_FW(N+1:2*N);
        FW_bit = qamdemod(sc_FW, 2^mod_ord,'OutputType','bit');
        ber_FW(snr_pt) = ber_FW(snr_pt)+sum(FW_bit~=s_bit);    

        % FW DetNet;
        tic
        s_FWunfold = FW_DetNet(x_ini, HH, Hy, 20, FWpar0, FWpar1, FWpar2);
        t_FWunfold(snr_pt) = t_FWunfold(snr_pt) + toc;
        s_FWunfold = deQAM(s_FWunfold, cons);
        sc_FWunfold = s_FWunfold(1:N)+sqrt(-1)*s_FWunfold(N+1:2*N);
        FWunfold_bit = qamdemod(sc_FWunfold, 2^mod_ord,'OutputType','bit');
        ber_FWunfold(snr_pt) = ber_FWunfold(snr_pt)+sum(FWunfold_bit~=s_bit);    

        % HoT-ADMM
        tic
        s_ADMM = HoT_ADMM(HH, Hy, Lf, x_ini_box);
        t_ADMM(snr_pt) = t_ADMM(snr_pt) + toc;
        s_ADMM = deQAM(s_ADMM, cons);
        sc_ADMM = s_ADMM(1:N)+sqrt(-1)*s_ADMM(N+1:2*N);
        ADMM_bit = qamdemod(sc_ADMM, 2^mod_ord,'OutputType','bit');
        ber_ADMM(snr_pt) = ber_ADMM(snr_pt)+sum(ADMM_bit~=s_bit); 

        % ADMM DetNet;
        tic
        s_ADMMunfold = ADMM_DetNet(x_ini, HH, Hy, 20, ADMMpar0, ADMMpar1, ADMMpar2);
        t_ADMMunfold(snr_pt) = t_ADMMunfold(snr_pt) + toc;  
        s_ADMMunfold = deQAM(s_ADMMunfold, cons);
        sc_ADMMunfold = s_ADMMunfold(1:N)+sqrt(-1)*s_ADMMunfold(N+1:2*N);
        ADMMunfold_bit = qamdemod(sc_ADMMunfold, 2^mod_ord,'OutputType','bit');
        ber_ADMMunfold(snr_pt) = ber_ADMMunfold(snr_pt)+sum(ADMMunfold_bit~=s_bit);
    
    end
    fprintf("SNR:%d dB done \n", SNR(snr_pt))
end

%compute the SER
ber_NI = ber_NI/(max_no_simulations*N*mod_ord);
ber_Box = ber_Box/(max_no_simulations*N*mod_ord);
ber_PG = ber_PG/(max_no_simulations*N*mod_ord);
ber_PGunfold = ber_PGunfold/(max_no_simulations*N*mod_ord);
ber_FW = ber_FW/(max_no_simulations*N*mod_ord);
ber_FWunfold = ber_FWunfold/(max_no_simulations*N*mod_ord);
ber_ADMM = ber_ADMM/(max_no_simulations*N*mod_ord);
ber_ADMMunfold = ber_ADMMunfold/(max_no_simulations*N*mod_ord);



%compute average running time
t_NI = t_NI/(max_no_simulations);
t_Box = t_Box/(max_no_simulations);
t_PG = t_PG/(max_no_simulations);
t_PGunfold = t_PGunfold/(max_no_simulations);
t_FW = t_FW/(max_no_simulations);
t_FWunfold = t_FWunfold/(max_no_simulations);
t_ADMM = t_ADMM/(max_no_simulations);
t_ADMMunfold = t_ADMMunfold/(max_no_simulations);


H1 = figure;
semilogy(SNR,ber_Box,'--h', 'Linewidth',1.5,'markers',8);hold on;
semilogy(SNR,ber_PG,'-h', 'Linewidth',1.5,'markers',8);hold on;
semilogy(SNR,ber_FW,'-^b', 'Linewidth',1.5,'markers',8);hold on;
semilogy(SNR,ber_ADMM,'-<', 'Linewidth',1.5,'markers',8);hold on;
semilogy(SNR,ber_PGunfold,'-dc', 'Linewidth',1.5,'markers',8);hold on;
semilogy(SNR,ber_FWunfold,'-ok', 'Linewidth',1.5,'markers',8);hold on;
semilogy(SNR,ber_ADMMunfold,'->', 'Linewidth',1.5,'markers',8);hold on;
semilogy(SNR,ber_NI,'-r', 'Linewidth',1.5,'markers',8);hold on;

% title("M="+num2str(2*M)+", N="+num2str(2*N))
legend('\fontsize{12}Box','\fontsize{12}HoT-PG','\fontsize{12}HoT-FW','\fontsize{12}HoT-ADMM','\fontsize{12}PG DetNet','\fontsize{12}FW DetNet','\fontsize{12}ADMM DetNet','\fontsize{12}lower bound' )
xlabel('SNR / (dB)')
ylabel('Bit Error Rate (BER)')
axis([0,20,1e-5,1])
hold on
grid on




