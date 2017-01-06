
isMatlab = exist('OCTAVE_VERSION', 'builtin') == 0;
if (~isMatlab)
    pkg load statistics;
    rand('seed',1);
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
                      [2,3,1;70,50,80],
                      ones(2,3),
                      repmat([0.99,0.01;0.1,0.9],[1,1,3]));

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
