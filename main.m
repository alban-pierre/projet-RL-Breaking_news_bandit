% Main file, run it to create arms and run RL algorithms (TS, UCB, KNN_UCB)


isMatlab = exist('OCTAVE_VERSION', 'builtin') == 0;
if (~isMatlab)
    pkg load statistics;
%    rand('seed',1);
end

addpath('knn/');
addpath('em_gaussian/');

gaussian = 1; 


% One arm can became hot at the same time
MAB1 = oneHotArm(repmat(gaussian,2,3),[2,3,1;70,50,80],ones(2,3),ones(1,3)/100,ones(1,3)/10);

% Many arms can became hot at the same time, more than 10 times slower
MAB2 = severalHotArms(ones(1,3),
                      repmat(gaussian,2,3),
                      [2,3,1;70,50,80],
                      ones(2,3),
                      repmat([0.99,0.01;0.1,0.9],[1,1,3]));

% I forgot to update the means for MAB2, they were different from MAB1
%MAB2 = severalHotArms(ones(1,3),repmat(gaussian,2,3),[0.2,0.3,0.1;0.7,0.5,0.8],ones(2,3),repmat([0.99,0.01;0.1,0.9],[1,1,3]));

% Plot arms rewards, and the computation time of one arm sampling
if (false)
    tt = time();
    s = zeros(3,1000);
    for i=1:1000
        s(1,i) = MAB1.sample(1);
        s(2,i) = MAB1.sample(2);
        s(3,i) = MAB1.sample(3);
    end
    time() - tt
    figure(1);
    plot(s');

    tt = time()
    s = zeros(3,1000);
    for i=1:1000
        s(1,i) = MAB2.sample(1);
        s(2,i) = MAB2.sample(2);
        s(3,i) = MAB2.sample(3);
    end
    time() - tt
    figure(2);
    plot(s');
end

% Plots the expectation of rewards in function of the time since the last draw
if (false)
    clear means;
    means{1} = [2,70];
    means{2} = [3,50];
    means{3} = [1,80];
    clear tr;
    tr{1} = [0.99,0.01;0.1,0.9]';
    tr{2} = [0.99,0.01;0.1,0.9]';
    tr{3} = [0.99,0.01;0.1,0.9]';
    plot_expectations(tr, means, 100);
end



tmax = 10000; % KNN_UCB works efficiently for t > 2000
ntests = 10;

allrew = zeros(3,tmax);
%tt = time();
for i=1:ntests
    [rew, draws] = UCB(tmax, MAB1);
    allrew(1,:) = allrew(1,:) + rew;
    %[rew, draws] = TS(tmax, MAB1);
    %allrew(1,:) = allrew(1,:) + rew;
    %[rew, draws] = TSvar(1000, MAB1);
    %allrew(1,:) = allrew(1,:) + rew;
    %[rew, draws] = GM(tmax, MAB1);
    %allrew(1,:) = allrew(1,:) + rew;
    [rew, draws] = KNN_UCB_NEW(tmax, MAB1);
    allrew(3,:) = allrew(3,:) + rew;
    %[rew, draws] = KNN_UCB_OLD(tmax, MAB1);
    %allrew(1,:) = allrew(1,:) + rew;
    %[rew, draws] = UCB(tmax, MAB1);
    %allrew(1,:) = allrew(1,:) + rew;
    [rew, draws, hot_expected, hot_real] = UCB_BN(tmax, MAB1);
    allrew(2,:) = allrew(2,:) + rew;
    if(i==ntests)
        figure;
        subplot(3,1,1);
        stairs(hot_real');
        subplot(3,1,2);
        stairs(hot_expected','r');
        subplot(3,1,3);
        stairs(hot_real'+1/10); hold on;
        stairs(hot_expected','r');
    end
    fprintf(2,'.');
end
%time() - tt
allrew = allrew./ntests;



%save('results/ucb_bn_10000_100.mat', 'allrew');

figure;
plot(1:tmax, cumsum(allrew(1,:)), '.k');
hold on;
plot(1:tmax, cumsum(allrew(2,:)),'.m');
plot(1:tmax, cumsum(allrew(3,:)),'.g');



if (false) % Plots stored results for each algorithm
    % One hot arms
    ts = load('results/ts_10000_100.mat');
    ucb = load('results/ucb_10000_100.mat');
    knn_ucb_old = load('results/knn_ucb_old_10000_10.mat');
    knn_ucb_new = load('results/knn_ucb_new_10000_10.mat');
    gm = load('results/gm_10000_1.mat');
    ucb_bn = load('results/ucb_bn_10000_100.mat');
    figure;
    plot(1:10000, cumsum(ts.allrew), '.b');
    hold on;
    plot(1:10000, cumsum(ucb.allrew), '.k');
    plot(1:10000, cumsum(knn_ucb_old.allrew), '.r');
    plot(1:10000, cumsum(knn_ucb_new.allrew), '.g');
    plot(1:10000, cumsum(gm.allrew), '.c');
    plot(1:10000, cumsum(ucb_bn.allrew), '.m');
    title('One hot arm model');
    xlabel('Iterations');
    ylabel('Cumulative rewards');
    legend('TS', 'UCB', 'KNN-0', 'KNN-1', 'GM', 'BN', 'location', 'southeast');
    figure;
    plot(1:1000, cumsum(ts.allrew(1,1:1000)), '.b');
    hold on;
    plot(1:1000, cumsum(ucb.allrew(1,1:1000)), '.k');
    plot(1:1000, cumsum(knn_ucb_old.allrew(1,1:1000)), '.r');
    plot(1:1000, cumsum(knn_ucb_new.allrew(1,1:1000)), '.g');
    plot(1:1000, cumsum(gm.allrew(1,1:1000)), '.c');
    plot(1:1000, cumsum(ucb_bn.allrew(1,1:1000)), '.m');
    title('One hot arm model');
    xlabel('Iterations');
    ylabel('Cumulative rewards');
    legend('TS', 'UCB', 'KNN-0', 'KNN-1', 'GM', 'BN', 'location', 'southeast');
    
    % Multiple hot arms
    ts = load('results/ts_m_10000_100.mat');
    ucb = load('results/ucb_m_10000_100.mat');
    knn_ucb_old = load('results/knn_ucb_old_m_10000_10.mat');
    knn_ucb_new = load('results/knn_ucb_new_m_10000_10.mat');
    gm = load('results/gm_m_10000_1.mat');
    figure;
    plot(1:10000, cumsum(ts.allrew), '.b');
    hold on;
    plot(1:10000, cumsum(ucb.allrew), '.k');
    plot(1:10000, cumsum(knn_ucb_old.allrew), '.r');
    plot(1:10000, cumsum(knn_ucb_new.allrew), '.g');
    plot(1:10000, cumsum(gm.allrew), '.c');
    title('Several hot arms model');
    xlabel('Iterations');
    ylabel('Cumulative rewards');
    legend('TS', 'UCB', 'KNN-0', 'KNN-1', 'GM', 'location', 'southeast');
    figure;
    plot(1:1000, cumsum(ts.allrew(1,1:1000)), '.b');
    hold on;
    plot(1:1000, cumsum(ucb.allrew(1,1:1000)), '.k');
    plot(1:1000, cumsum(knn_ucb_old.allrew(1,1:1000)), '.r');
    plot(1:1000, cumsum(knn_ucb_new.allrew(1,1:1000)), '.g');
    plot(1:1000, cumsum(gm.allrew(1,1:1000)), '.c');
    title('Several hot arms model');
    xlabel('Iterations');
    ylabel('Cumulative rewards');
    legend('TS', 'UCB', 'KNN-0', 'KNN-1', 'GM', 'location', 'southeast');
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Computation time : (One hot arm)
    % TS -------- :    7.1236s pour tmax = 10000
    % UCB ------- :    2.1950s pour tmax = 10000
    % KNN_UCB_OLD :   29.452s  pour tmax = 10000    % By old I mean the resize in [0-1] is made after knn
    % KNN_UCB_NEW :   25.389s  pour tmax = 10000    % By new I mean the resize in [0-1] is made before knn
    % GM -------- : 1000.5s    pour tmax = 10000
    % BN -------- :    3.0577s pour tmax = 10000


    % Computation time : (Several hot arm)
    % TS -------- :   31.727s pour tmax = 10000
    % UCB ------- :   26.200s pour tmax = 10000
    % KNN_UCB_OLD :   52.046s pour tmax = 10000    % By old I mean the resize in [0-1] is made after knn
    % KNN_UCB_NEW :   49.382s pour tmax = 10000    % By new I mean the resize in [0-1] is made before knn
    % GM -------- : 1005.4s   pour tmax = 10000
