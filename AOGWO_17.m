%--------------------------------------------------------------------------
% Title: An Adaptive Oppositional Grey Wolf Optimizer for Complex Engineering Problems
% Authors: Othman Waleed Khalid, Nor Ashidi Mat Isa, Karrar Mohsin Alwan, 
%          Sew Sun Tiang, Ahbishek Sharma, Tarek Berghout, Jun-Jiat Tiang, Wei Hong Lim
% Year: 2026
%  
% Description: 
% This function implements the Adaptive Oppositional Grey Wolf Optimizer (AOGWO).
% The proposed approach addresses limitations of the standard GWO by incorporating:
% 1. Hybrid Opposition-Based Learning (OBL) Initialization to ensure maximum initial spatial diversity.
% 2. Adaptive Cosine Control strategy paired with a decaying Jumping Rate (Jr) to balance search phases.
% 3. Levy flight perturbations to prevent stagnation and boost global exploration.
% 4. Selective Leading Opposition (SLO) mechanism applied exclusively to the Alpha leader to prevent elite traps.
%--------------------------------------------------------------------------

function AOGWO = AOGWO_17(opts)
    % ---------------- Parameters Extraction ----------------
    if isfield(opts, 'maxIter'), max_Iter = opts.maxIter; 
    else max_Iter = 200; 
    end
    if isfield(opts, 'maxFES'), max_FES = opts.maxFES; 
    else max_FES = 100000; 
    end
    if isfield(opts, 'dimension'), dim = opts.dimension; 
    else dim = 100; 
    end
    if isfield(opts, 'testFunction'), func = opts.testFunction; 
    else func = 1; 
    end

    % ---------------- Algorithm Settings ----------------
    N = 50; % Population size 
    lb = 0; % Lower bound of search space 
    ub = 1;  % Upper bound of search space  
    fitcount = 0; % Function evaluation counter
    
    % Initialize Wolf hierarchy leaders
    Alpha_pos = zeros(1, dim); Alpha_score = inf;
    Beta_pos = zeros(1, dim);  Beta_score = inf;
    Delta_pos = zeros(1, dim); Delta_score = inf;

    % ---------------- 1. Hybrid OBL Initialization ----------------
    % Replaces standard random initialization to systematically and evenly 
    % distribute candidate solutions.
    X = lb + (ub - lb) * rand(N, dim);
    for i = 1:N
        % Opposite Position (OP) calculation
        OP = ((ub - lb) * rand(1, dim)) + lb - X(i, :);
        % Randomized Opposite (RO)
        RO = rand * OP;
        % Dynamic Opposite (DO) jump implementation
        DO = X(i, :) + rand * (RO - X(i, :));
        
        % Ensure the initialized wolves remain strictly within bounds
        X(i, :) = max(min(DO, ub), lb);
    end
    
    % Initialize variables for recording optimization performance
    curve = zeros(1, max_Iter);
    t = 1;

    % ---------------- Main Optimization Loop ----------------
    while t <= max_Iter && fitcount < max_FES
        % Fitness evaluation and Hierarchy Update
        for i = 1:N
            % Enforce search space boundaries 
            X(i, :) = max(min(X(i, :), ub), lb);
            
            % Evaluate objective function
            fitness = cec17_func(X(i, :)', func);
            fitcount = fitcount + 1;
            
            % Update Alpha, Beta, and Delta wolves based on fitness
            if fitness < Alpha_score
                Alpha_score = fitness; 
                Alpha_pos = X(i, :);
            elseif fitness < Beta_score
                Beta_score = fitness; 
                Beta_pos = X(i, :);
            elseif fitness < Delta_score
                Delta_score = fitness; 
                Delta_pos = X(i, :);
            end
        end

        % ---------------- 2. Adaptive Cosine Control & Jumping Rate ----------------
        % Instead of rigid linear decay, a cosine function dynamically adapts parameter 'a'
        a = 2 * cos((pi/2) * (t/max_Iter)); 
        
        % Jumping Rate (Jr) dictating exploration likelihood (Linear decrease over iterations)
        Jr = 0.5 * (1 - t/max_Iter); 

        % ---------------- Position Update Strategy ----------------
        for i = 1:N
            if rand < Jr
                % ---------------- 3. Levy Flight Perturbation ----------------
                % Broad exploration tactic executed when Jr conditions are met.
                % Uses Levy steps to execute severe jumps, breaking potential stagnation.
                X(i, :) = Alpha_pos - a * abs(rand * Alpha_pos - X(i, :)) + rand(1, dim) .* Levy(dim);
            else
                % ---------------- Standard GWO Exploitation ----------------
                % Hierarchical updating guided by Alpha, Beta, and Delta utilizing the adaptive 'a'
                for j = 1:dim
                    % Interaction with Alpha leader
                    r1 = rand(); r2 = rand();
                    A1 = 2*a*r1 - a; C1 = 2*r2;
                    D_alpha = abs(C1*Alpha_pos(j) - X(i,j));
                    X1 = Alpha_pos(j) - A1*D_alpha;

                    % Interaction with Beta leader
                    r1 = rand(); r2 = rand();
                    A2 = 2*a*r1 - a; C2 = 2*r2;
                    D_beta = abs(C2*Beta_pos(j) - X(i,j));
                    X2 = Beta_pos(j) - A2*D_beta;

                    % Interaction with Delta leader
                    r1 = rand(); r2 = rand();
                    A3 = 2*a*r1 - a; C3 = 2*r2;
                    D_delta = abs(C3*Delta_pos(j) - X(i,j));
                    X3 = Delta_pos(j) - A3*D_delta;

                    % Compute the final updated position vector
                    X(i,j) = (X1 + X2 + X3) / 3;
                end
            end
        end

        % ---------------- 4. Selective Leading Opposition (SLO) ----------------
        % A refined, targeted JOS strategy applied specifically to the Alpha.
        % This forces the best solution to occasionally search its inverse space 
        % to drastically prevent elite entrapment.
        if rand < 0.5
            current_min = min(X);
            current_max = max(X);
            
            % Compute the opposite of the Alpha using real-time dynamic bounds
            Opp_Alpha = current_min + current_max - Alpha_pos;
            Opp_Alpha = max(min(Opp_Alpha, ub), lb); % Restrict to bounds
            
            % Evaluate the theoretical opposite leader
            f_opp = cec17_func(Opp_Alpha', func);
            fitcount = fitcount + 1;
            
            %   Accept the opposite position if it improves fitness
            if f_opp < Alpha_score
                Alpha_score = f_opp;
                Alpha_pos = Opp_Alpha;
            end
        end

        % Record the convergence performance for the current iteration
        curve(t) = Alpha_score;
        t = t + 1;
        
        % Terminate early if maximum function evaluations are exceeded
        if fitcount >= max_FES, break; end
    end

    % Package outputs
    AOGWO.bestFitness = Alpha_score;
    AOGWO.convergenceCurve = curve;
end

% ---------------- Levy Flight Step Generator ----------------
% Provides heavy-tailed step generation necessary to simulate aggressive spatial jumps
function L = Levy(dim)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, dim) * sigma;
    v = randn(1, dim);
    step = u ./ abs(v).^(1 / beta);
    L = 0.01 * step;  
end