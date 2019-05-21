% Multi object kalman filter
clear;
close all;
clc

%% Loading video

formatSpec = ("videos/pos/%s.avi");
name = '17ntp+-00';
str = sprintf(formatSpec,name);
video = importdata(str);
%% Preprocessing and detection
X = cell(1,length(video));
Y = cell(1,length(video));
for i=1:length(video)
    binary_filter = imbinarize(video(i).cdata,0.5);
    se = strel('diamond', 5);
    image_dia = imdilate(binary_filter,se);
    image_erod = imerode(image_dia,se);
    CC = bwconncomp(image_erod, 8);
    
    s = regionprops(CC,'centroid','Area','Eccentricity');
    centroids = cat(1,s.Centroid);
    if i==1
        median = median([s.Area]);
        Amean = mean([s.Area]);
    end
    idx = find([s.Area] < median*10); %% removing object with more than median*10 pixles (Debris)
    s1 = s(idx,:);
    idx1 = find([s1.Area]< Amean*2.5 & [s1.Area]> Amean*0.5);
    s2 = s1(idx1,:);
    centroids = centroids(idx,:);
    X{i} = centroids(:,2);
    Y{i} = centroids(:,1);
        imshow(video(i).cdata);
        hold on
       plot(Y{i},X{i},'go');
end
%% define main variables for kalman filter
dt = 1;  %our sampling rate
S_frame = 1; %Start frame

u = 0; % define acceleration magnitude to start
noise_mag = 1.5; %process noise: the variability in how fast the cells is speeding up (stdv of acceleration: meters/sec^2)
tkn_x = .15;  %measurement noise in the horizontal direction (x axis).
tkn_y = .15;  %measurement noise in the horizontal direction (y axis).
Ez = [tkn_x 0; 0 tkn_y];
Ex = [dt^4/4 0 dt^3/2 0; ...
    0 dt^4/4 0 dt^3/2; ...
    dt^3/2 0 dt^2 0; ...
    0 dt^3/2 0 dt^2].*noise_mag^2; % covariance matrix
P = Ex; % estimate of initial position variance (covariance matrix)

no_trk_list = cell(1,length(video));

%% Define update equations in 2-D (Coefficent matrices): A physics based model for where we expect the objects to be [state transition (state + velocity)] + [input control (acceleration)]
A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]; %state update matrice
B = [(dt^2/2); (dt^2/2); dt; dt];
C = [1 0 0 0; 0 1 0 0];  %this is our measurement function C, that we apply to the state estimate Q to get our expect next/new measurement




%% initize result variables
Q_loc_meas = []; % the detecions  extracted by the detection algorithm
%% initize estimation variables for two dimensions
Q= [X{S_frame} Y{S_frame} zeros(length(X{S_frame}),1) zeros(length(X{S_frame}),1)]';
Q_estimate = nan(4,20000);
Q_estimate(:,1:size(Q,2)) = Q;  %estimate of initial location estimation of where the objects are(what we are updating)
Q_loc_estimateY = nan(length(video),20000); %  position estimate
Q_loc_estimateX= nan(length(video),20000); %  position estimate
P_estimate = P;  %covariance estimator
strk_trks = zeros(1,20000);  %counter of how many strikes a track has gotten
nD = size(X{S_frame},1); %initize number of detections
nF =  find(isnan(Q_estimate(1,:))==1,1)-1 ; %initize number of track estimates

%% for each frame
for t = S_frame:length(video)
    % make the given detections matrix
    Q_loc_meas = [X{t} Y{t}];
    
    %% do the kalman filter
    % Predict next state of the objects with the last state and predicted motion.
    nD = size(X{t},1); %set new number of detections
    for F = 1:nF
        Q_estimate(:,F) = A * Q_estimate(:,F) + B * u;
    end
    
    %predict next covariance
    P = A * P* A' + Ex;
    
    
    
    %% now we assign the detections to estimated track positions
    %make the distance (cost) matrice between all pairs rows = tracks, coln =
    %detections
    est_dist = pdist([Q_estimate(1:2,1:nF)'; Q_loc_meas]);
    est_dist = squareform(est_dist); %make square
    est_dist = est_dist(1:nF,nF+1:end) ; %limit to just the tracks to detection distances
    
    [asgn, cost] = assignmentoptimal(est_dist); %do the assignment with hungarian algorithm
    asgn = asgn';
    
    % ok, now we check for tough situations and if it's tough, just go with estimate and ignore the data
    %make asgn = 0 for that tracking element
    
    %check 1: is the detection far from the observation? if so, reject it.
    rej = nan(1,nF);
    for F = 1:nF
        if asgn(F) > 0
            rej(F) =  est_dist(F,asgn(F)) < 15 ;
        else
            rej(F) = 0;
        end
    end
    asgn = asgn.*rej;
    
    
    %% apply the assingment to the update
    
    % Kalman Gain
    K = P*C'/(C*P*C'+Ez);
    
    k = 1;
    for F = 1:length(asgn)
        if asgn(F) > 0
            Q_estimate(:,k) = Q_estimate(:,k) + K * (Q_loc_meas(asgn(F),:)' - C * Q_estimate(:,k));
        end
        k = k + 1;
    end
    
    % update covariance estimation.
    P =  (eye(4)-K*C)*P;
    
    %% Store data
    Q_loc_estimateX(t,1:nF) = Q_estimate(1,1:nF);
    Q_loc_estimateY(t,1:nF) = Q_estimate(2,1:nF);
    
    %ok, now that we have our assignments and updates, lets find the new detections and
    %lost trackings
    
    %find the new detections. basically, anything that doesn't get assigned
    %is a new tracking
    %    new_trk = [];
    new_trk = Q_loc_meas(~ismember(1:size(Q_loc_meas,1),asgn),:)';
    if ~isempty(new_trk)
        Q_estimate(:,nF+1:nF+size(new_trk,2))=  [new_trk; zeros(2,size(new_trk,2))];
        nF = nF + size(new_trk,2);  % number of track estimates with new ones included
    end
    
    
    %give a strike to any tracking that didn't get matched up to a
    %detection
    no_trk_list{t} =  find(asgn==0);
    if ~isempty(no_trk_list{t})
        strk_trks(no_trk_list{t}) = strk_trks(no_trk_list{t}) + 1;
    end
    
    
    %if a track has a strike greater than 6, delete the tracking. i.e.
    %make it nan first vid = 3
    
    bad_trks = find(strk_trks > 6);
    Q_estimate(:,bad_trks) = NaN;
    
    %% Plotting the image
    %     clf;
    %     img = video(t).cdata;
    %     imshow(img);
    %     hold on;
    %     plot(Y{t}(:),X{t}(:),'or'); % the actual tracking
    %     T = size(Q_loc_estimateX,2);
    %     Ms = [1 1]; %marker sizes
    %     c_list = ['r' 'b' 'g' 'c' 'm' 'y'];
    %     for Dc = 1:nF
    %         if ~isnan(Q_loc_estimateX(t,Dc))
    %             Sz = mod(Dc,2)+1; %pick marker size
    %             Cz = mod(Dc,6)+1; %pick color
    %
    %             st = t-1;
    %
    %             tmX = Q_loc_estimateX(t-st:t,Dc);
    %             tmY = Q_loc_estimateY(t-st:t,Dc);
    %             plot(tmY,tmX,'.-','markersize',Ms(Sz),'color',c_list(Cz),'linewidth',1)
    %         end
    %     end
    %     disp(t)
    %         pause
end
avgCellCount = sum(cellfun('length',X))/length(X);
CON = (avgCellCount/10.41)*10^6;

for i=1:length(no_trk_list)
    Q_loc_estimateX(i,no_trk_list{i}) = NaN;
    Q_loc_estimateY(i,no_trk_list{i}) = NaN;
end

new_locX = Q_loc_estimateX;
new_locY = Q_loc_estimateY;



new_locX(~any(isnan(new_locX),2),:) = [];
new_locY(:,any(isnan(new_locY), 1)) = [];
Q_loc_estimateX(:,any(isnan(Q_loc_estimateX(1,:)), 1)) = [];
Q_loc_estimateY(:,any(isnan(Q_loc_estimateY(1,:)), 1)) = [];

%% Parameter estimation
Q_loc_estimateX = Q_loc_estimateX';
Q_loc_estimateY = Q_loc_estimateY';
loc_cellX = cell(1,length(Q_loc_estimateX(1,:)));
loc_cellY = cell(1,length(Q_loc_estimateY(1,:)));

for i=1:length(Q_loc_estimateX(:,1))
    loc_cellX{i} = rmmissing(Q_loc_estimateX(i,:));
    loc_cellY{i} = rmmissing(Q_loc_estimateY(i,:));
end
for i=length(Q_loc_estimateX(:,1)):-1:1
    if length(loc_cellX{i})<5
        loc_cellX(i) = [];
        loc_cellY(i) = [];
    end
end
for i=1:length(loc_cellX)
    moving_mean = movmean([loc_cellX{i}',loc_cellY{i}'],5);
    
    for j=1:length(loc_cellX{1,i})-1
        p1 = [loc_cellX{1,i}(1,j); loc_cellY{1,i}(1,j)];
        p2 = [loc_cellX{1,i}(1,j+1); loc_cellY{1,i}(1,j+1)];
        d(i,j) = norm(p1 - p2);
    end
    for j=3:length(loc_cellX{1,i})-1
        p1_mean = [moving_mean(j-1,1); moving_mean(j-1,2)];
        p2_mean = [moving_mean(j,1); moving_mean(j,2)];
        d_mean(i,j) = norm(p1_mean - p2_mean);
    end
    
    VSL(i,1)=norm(([loc_cellX{1,i}(1,1);loc_cellY{1,i}(1,1)])-([loc_cellX{1,i}(1,end);loc_cellY{1,i}(1,end)]));
    VCL(i,1) = sum(d(i,:));
    VAP(i,1) = sum(d_mean(i,:)+d(i,1)+d(i,end));
    ALH(i,1) = sqrt(sum((VAP(i,1) - VCL(i,1)).^2));
end

LIN = VSL(:,1)./VCL(:,1);
WOB = VAP./VCL;
STR = VSL./VAP;


LIN(~isfinite(LIN)) = 0;
WOB(~isfinite(WOB)) = 0;
STR(~isfinite(STR)) = 0;

FP = zeros(length(VSL),7);
FP(:,1) = VSL;
FP(:,2) = VCL;
FP(:,3) = VAP;
FP(:,4) = LIN;
FP(:,5) = WOB;
FP(:,6) = STR;
FP(:,7) = ALH;


%% Plot trackes based on LIN
% figure;
% xy = video(25).cdata';
% imshow(xy,[]); hold on
% for i=1:length(loc_cellX)
%     if (FP(i,4) > 0.8 && FP(i,1) > 60)
%         plot(loc_cellX{:,i},loc_cellY{:,i},'r-','MarkerSize',10);
%         drawnow;
%         hold on;
%         %     pause(0.3)
%     else
%         plot(loc_cellX{:,i},loc_cellY{:,i},'g-','MarkerSize',10);
%         drawnow;
%         hold on;
%     end
% end
index = find(LIN>0.8 & VSL>60);
FP(index,:) = [];
loc_cellX(:,index) = [];
loc_cellY(:,index) = [];
ER = length(index)+length(bad_trks);
%%
formatSpec = ("%s.mat");
str = sprintf(formatSpec,name);
save(str,'FP','CON','ER')
figure;
subplot(311)
histogram(FP(:,6),100)
subplot(312)
histogram(FP(:,4),100)
subplot(313)
histogram(FP(:,5),100)