
isMatlab = exist('OCTAVE_VERSION', 'builtin') == 0;
if (~isMatlab)
    pkg load statistics;
    rand('seed',1);
end


% Many arms can became hot at the same time
MAB1 = {armGaussian(1, [2,7], [1,1], [100,1;10,1]),
        armGaussian(1, [3,5], [1,1], [100,1;10,1]),
        armGaussian(1, [1,8], [1,1], [100,1;10,1])}

% One arms can became hot at the same time
MAB2 = severalArmGaussian([2,3,1;7,5,8], ones(2,3), ones(1,3)/100, ones(1,3)/10);


MAB1{1}.sample()
MAB1{2}.sample()
MAB1{3}.sample()
MAB1{2}.sample()
MAB1{1}.sample()
MAB1{3}.sample()

MAB2.sample(1)
MAB2.sample(2)
MAB2.sample(3)
MAB2.sample(2)
MAB2.sample(1)
MAB2.sample(3)
