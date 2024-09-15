clear all;
close all;
clc;
%% Read Data
% Specify the folder where the files are
myFolder = 'D:\pr1\data';
% Get a list of all EDF files in the folder
filePattern = fullfile(myFolder, '*.edf'); % Pattern for EDF files
theFiles = dir(filePattern);
% Initialize large matrix
largeMatrix = [];

% file reading 24
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(myFolder, baseFileName);
    fprintf('Now reading %s\n', fullFileName);
    
        % Get file info using edfinfo
        fileInfo = edfinfo(fullFileName);

        % Display fileInfo structure to understand its fields
        disp(fileInfo);
        
        % Read EDF file using edfread
        tt = edfread(fullFileName); 
        % Get the channel names
        channelNames = tt.Properties.VariableNames;
        
        % Extract data for each channel
        for i = 1:length(channelNames)
            channelName = channelNames{i};
            channelData = tt.(channelName);
            for iii=1:1:300
                dat=channelData{iii, 1};
                largeMatrix=[largeMatrix; dat'];  
            end
        end
        
end
%% 
clear all;
load("large.mat")
%%
temp1 = [];
 temp2 = [];
 for i = 1:2:288000
     temp1 = [];  
     for j = i+0:1:i+1
         temp1 = [temp1 largeMatrix(j,:)];  
     end
     temp2 = [temp2; temp1];  
 end

%% Values to be removed
% Initialize the row indices to be removed
rows_to_remove = [];

% Define the ranges of rows to be removed within each 6000-row segment
remove_ranges = [151:300, 3901:4050, 4801:4950, 5401:6000];

% Iterate over the 24 segments of 6000 rows each
for k = 0:23
    % Calculate the base index for the current 6000-row segment
    base_idx = k * 6000;
    
    % Add the indices to be removed for the current segment
    rows_to_remove = [rows_to_remove, base_idx + remove_ranges];
end

% Remove the rows from temp2
temp2(rows_to_remove, :) = [];

% Display the new size of temp2 to verify the rows have been removed
disp(size(temp2));



%%
%SWD
clear all;
close all;

 load('finalmatrix.mat');
%%
clc

modes=[];
for rr=64252:1:size(temp2,1)
    disp(rr)
x    = temp2(rr, :);
x    = (x - mean(x)) / max(abs(x));
fs   = 1000; 
dt   = 1 / fs;
L    = length(x);
t    = 0:dt:(L - 1) * dt;
nfft = 2 ^ nextpow2(L);
X    = abs(fft(x, nfft)) / L;
f    = (fs / 2) * linspace(0, 1, nfft/2);

% figure;
% subplot(1, 2, 1); plot(t, x); 
% subplot(1, 2, 2); plot(f, X(1:end/2));

% bandpass filtering
[b, a] = butter(5, [4 31.5] / (fs / 2), 'bandpass');
x_filt = filtfilt(b, a, x);
nfft   = 2 ^ nextpow2(length(x_filt));
X_filt = abs(fft(x_filt, nfft)) / nfft;
f      = (fs / 2) * linspace(0, 1, nfft/2);

figure; 
subplot(1, 2, 1); plot(t, x_filt); 
subplot(1, 2, 2); plot(f, X_filt(1:end/2));

% downsamping
q     = 4; 
fs2   = fs / q;
dt2   = 1 / fs2;
x_dec = resample(x_filt, 1, q); 
L_dec = length(x_dec);
nfft  = 2 ^ nextpow2(L_dec);
X_dec = abs(fft(x_dec, nfft)) / L_dec; 
t2    = 0:dt2:(length(x_dec) - 1) * dt2;
f2    = (fs2 / 2) * linspace(0, 1, nfft/2);

figure; 
subplot(1, 2, 1); plot(t2, x_dec); 
subplot(1, 2, 2); plot(f2, X_dec(1:end/2));
% ******************* SWD execution *******************
L1    = length(x_dec);
s_SWD = [x_dec, zeros(1,100)];  % zero-padding
L2    = length(s_SWD);

welch_window      = round(L2 / 16);
welch_no_overlap  = round(welch_window / 2);
welch_nfft        = 2^nextpow2(L2);
param_struct      = struct('P_th', 0.2, ...
                           'StD_th', 0.2, ...
                           'Welch_window', welch_window, ...
                           'Welch_no_overlap', welch_no_overlap, ...
                           'Welch_nfft', welch_nfft);

y_SWD_res = SwD(s_SWD, param_struct); disp("end")
size(y_SWD_res,2)
y_SWD = y_SWD_res.';

modes{rr}=y_SWD;
close all;
% ******************* Plot results *******************t2
end
%% Step 3:feature extraction

%% Step 4:classfication (quntum)