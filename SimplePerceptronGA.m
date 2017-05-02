
%% READ IRIS DATA

fullData = csvread('irisDataEqual.dat');

sumOfErrors = 0;
for round = 1:100
fullData = fullData(randperm(length(fullData)),:);
numberOfTrainningSamples = 100;

% SPLIT IT IN TWO DIFFERENT ARRAYS
trainningData = fullData(1:numberOfTrainningSamples,:);
testData = fullData(numberOfTrainningSamples + 1:end,:);

%GETTING DATA WIDTH AND HEIGHT
width = size(trainningData);
width = width(2);
totalHeight = size(fullData);
totalHeight = totalHeight(1);
height = size(trainningData);
height = height(1);

global inputs
inputs = [repmat(-1, length(trainningData),1) trainningData(:, 1:width-1)];
global correctOutputs
correctOutputs = trainningData(:, width);

fitFunc = @fitnessOf;


% Start with the default options
options = optimoptions('ga');

% Modify options setting
options = optimoptions(options,'PopulationSize', 40);
options = optimoptions(options,'EliteCount', 4);
options = optimoptions(options,'ConstraintTolerance', 1e-30);
options = optimoptions(options,'CrossoverFcn', @crossoverarithmetic);
options = optimoptions(options,'Display', 'off');
[w,fval,exitflag,output,population,score] = ga(@fitnessOf,5,[],[],[],[],[],[],[],[],options);



%%
%==========================================
%             ZONA DE TESTE
%==========================================

% DIVIDE EM AMOSTRAS PARA TESTE E RESPOSTAS CORRETAS
testArray = [repmat(-1, length(testData), 1) testData(:,1:width-1)];
correctAnswers = testData(:, width)';

% MULTIPLY O ARRAY DE TESTE PELO W ENCONTRADO COM O ALGORITMO GENÉTICO
answers = testArray * w';

% APLICA A FUNÇÃO SINAL AOS VALORES ENCONTRADOS NO NEURÔNIO
for i = 1 : totalHeight - height
    answers(i) = sinalDe(answers(i));
end

% VERIFICA OS ERROS DA FASE DE TESTE
evaluation = answers - correctAnswers';
errorCount = sum(evaluation.^2);
sumOfErrors = sumOfErrors + errorCount;
round


end

sumOfErrors

function y = fitnessOf(weights)
global inputs
global correctOutputs
output = inputs*weights';
signal = sinalDe(output);
error = correctOutputs - signal';
y = (sum(error.^2) / length(error));

end


function y = sinalDe(x)
signal = [];

for i = 1:length(x)
    if x(i, 1) > 0
        signal = [signal 1];
    else
        signal = [signal 0];
    end
end

y = signal;


end
