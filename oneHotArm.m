classdef oneHotArm<handle

    % Class that gather many different arms for different states

    % It allows only one class to be hot (state > 1) at the same time
    % The status (hot or not) of all arms is updated when one arm is sampled

    % Input :
    % type   : (S*N)     : Type of each arm (gaussian only for now)
    % mean   : (S*N)     : Mean of each arm
    % v      : (S*N)     : Variance of each arm
    % p_to_H : ((S-1)*N) : Probabilities to become hot
    % p_to_N : ((S-1)*N) : Probabilities to come back to normal
        
    properties
        h      % (1*N)     : Which state is hot
        p_to_H % ((S-1)*N) : Probabilities to become hot
        p_to_N % ((S-1)*N) : Probabilities to come back to normal
        arms   % (S*N)     : Each arm
    end
    
    methods
        function self = oneHotArm(type, mean, v, p_to_H, p_to_N)
            self.h=ones(1,size(type,2));
            for n=1:size(type,2)
                for s=1:size(type,1)
                    switch type(s,n)
                        case 1 % gaussian
                            self.arms{s,n} = armGaussian(mean(s,n), v(s, n));
                        otherwise
                            self.arms{s,n}.sample = @() assert(false, 'Reached undefined state');
                    end
                end
            end
            self.p_to_H = p_to_H;
            assert(sum(p_to_H(:)) <= 1, 'Wrong transition probabilities');
            self.p_to_N = p_to_N;
        end
        
        function [reward, hotState] = sample(self, s)
            [maxh, hotState] = max(self.h,[],2);
            if (maxh == 1)
                hotState = 0;
            end
            reward = self.arms{self.h(1,s), s}.sample();
            if (sum(self.h,2) == size(self.h,2))
                if (rand(1) < sum(self.p_to_H(:)))
                    p = reshape(mnrnd(1,reshape(self.p_to_H,1,prod(size(self.p_to_H)))./sum(self.p_to_H(:))),size(self.p_to_H));
                    self.h(1,sum(p,1)*(1:size(p,2))') = 1+(1:size(p,1))*sum(p,2);
                    %self.h
                    %disp(['Entering hot state ', num2str(sum(p,1)*(1:size(p,2))')]);
                end
            else
                [~,n] = max(self.h,[],2);
                if (rand(1) < self.p_to_N(self.h(1,n)-1,n))
                    self.h(1,n) = 1;
                    %disp('Leaving hot state');
                end
            end
        end
        
        function NbArms = nbArms(self)
            NbArms = size(self.h,2);
        end
                       
    end
end
