package org.example.onnx;


import ai.onnxruntime.*;
import org.example.onnx.tokenizer.BertTokenizer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * https://github.com/microsoft/ai-edu
 *
 * https://github.com/microsoft/onnxruntime/tree/main/java
 * https://github.com/microsoft/onnxruntime/blob/main/java/README.md
 * <p>
 * https://blog.csdn.net/mzl87/article/details/109170580
 * https://github.com/keithpij/onnx-lab/blob/master/java_lab/src/App.java
 * https://juejin.cn/post/7266816831822741539
 * <p>
 * <p>
 * https://neuml.hashnode.dev/export-and-run-models-with-onnx
 * https://github.com/microsoft/onnxruntime/blob/main/java/src/test/java/ai/onnxruntime/InferenceTest.java
 * <p>
 * https://cloud.tencent.com/developer/article/2343710
 * https://github.com/microsoft/onnxruntime/issues/10142
 * https://www.jianshu.com/p/777d07037d00
 */
public class BertOnOnnx {

    static Map<String, String> categoryMap = new HashMap<>();

    static {
        categoryMap.put("0", "finance");
        categoryMap.put("1", "realty");
        categoryMap.put("2", "stocks");
        categoryMap.put("3", "education");
        categoryMap.put("4", "science");
        categoryMap.put("5", "society");
        categoryMap.put("6", "politics");
        categoryMap.put("7", "sports");
        categoryMap.put("8", "game");
        categoryMap.put("9", "entertainment");
    }

    public static void main(String[] args) throws OrtException {
        //Bert-Chinese-Text-Classification-Pytorch项目的 vocab.txt
        String vocabPath = "G:\\qzd\\JavaProject\\QZD_GROUP\\bird-query\\Bert-Chinese-Text-Classification-Pytorch\\bert_pretrain\\vocab.txt";
        //bert_to_onnx.py执行后的模型文件
        String modelPath = "G:\\qzd\\JavaProject\\QZD_GROUP\\bird-query\\Bert-Chinese-Text-Classification-Pytorch\\THUCNews\\saved_dict\\model.onnx";

        String query ="备考2012高考作文必读美文50篇(一)";

        BertTokenizer bertTokenizer = new BertTokenizer(vocabPath);
        Map<String, OnnxTensor> inputMap = bertTokenizer.tokenizeOnnxTensor(Arrays.asList(query));

        OrtEnvironment env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        try (OrtSession session = env.createSession(modelPath, options)) {
            // Load code not shown for brevity.
            // Run the inference
            try (OrtSession.Result results = session.run(inputMap)) {
                // Only iterates once
                for (Map.Entry<String, OnnxValue> r : results) {
                    OnnxValue resultValue = r.getValue();
                    OnnxTensor resultTensor = (OnnxTensor) resultValue;
                    int prediction = MaxProbability(resultTensor);
                    String category = categoryMap.get(String.valueOf(prediction));
                    System.out.println("Prediction: " + category);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    static int MaxProbability(OnnxTensor probabilities) throws OrtException {
        float[][] labelOutput = (float[][]) probabilities.getValue();

        float[] label = labelOutput[0];
        float max = -9999.9F;
        int maxIndex = -1;
        for (int i = 0; i < label.length; ++i) {
            float prob = label[i];
            if (prob > max) {
                max = prob;
                maxIndex = i;
            }
        }
        return maxIndex;

    }
}
