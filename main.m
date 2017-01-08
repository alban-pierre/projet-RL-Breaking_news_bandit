
isMatlab = exist('OCTAVE_VERSION', 'builtin') == 0;
if (~isMatlab)
    pkg load statistics;
%    rand('seed',1);
end

gaussian = 1;


% One arms can became hot at the same time
MAB1 = oneHotArm(repmat(gaussian,2,3),
                 [2,3,1;70,50,80],
                 ones(2,3),
                 ones(1,3)/100,
                 ones(1,3)/10);

% Many arms can became hot at the same time, more than 10 times slower
MAB2 = severalHotArms(ones(1,3),
                      repmat(gaussian,2,3),
                      [0.2,0.3,0.1;0.7,0.5,0.8],
                      ones(2,3),
                      repmat([0.99,0.01;0.1,0.9],[1,1,3]));

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

allrew = zeros(3,1000);

ntests = 10;

for i=1:ntests
    [rew, draws] = TS(1000, MAB1);
    allrew(1,:) = allrew(1,:) + rew;
    [rew, draws] = TSvar(1000, MAB1);
    allrew(2,:) = allrew(2,:) + rew;
    [rew, draws] = UCB(1000, MAB1);
    allrew(3,:) = allrew(3,:) + rew;
end
allrew = allrew./ntests;


figure;
plot(1:1000, cumsum(allrew(1,:)), 'b');
hold on;
plot(1:1000, cumsum(allrew(2,:)),'r');
plot(1:1000, cumsum(allrew(3,:)),'k');

