
isMatlab = exist('OCTAVE_VERSION', 'builtin') == 0;
if (~isMatlab)
    pkg load statistics;
    rand('seed',1);
end

gaussian = 1;

% Many arms can became hot at the same time
MAB1 = {armGaussian(1, [2,7], [1,1], [100,1;1,20]),
        armGaussian(1, [3,5], [1,1], [100,1;1,20]),
        armGaussian(1, [1,8], [1,1], [100,1;1,20])};

% One arms can became hot at the same time
MAB2 = severalArmGaussian([2,3,1;7,5,8], ones(2,3), ones(1,3)/100, ones(1,3)/20);

% Many arms can became hot at the same time
MAB1 = severalHotArms(ones(1,3),
                      repmat(gaussian,2,3),
                      [2,3,1;7,5,8],
                      ones(2,3),
                      repmat([0.99,0.01;0.1,0.9],[1,1,3]));


s = zeros(3,1000);
for i=1:1000
    s(1,i) = MAB1{1}.sample();
    s(2,i) = MAB1{2}.sample();
    s(3,i) = MAB1{3}.sample();
end
figure(1);
plot(s');

s = zeros(3,1000);
for i=1:1000
    s(1,i) = MAB2.sample(1);
    s(2,i) = MAB2.sample(2);
    s(3,i) = MAB2.sample(3);
end
figure(2);
plot(s');
