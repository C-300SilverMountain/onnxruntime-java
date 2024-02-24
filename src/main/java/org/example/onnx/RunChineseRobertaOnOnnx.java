package org.example.onnx;


import ai.onnxruntime.*;
import org.example.onnx.tokenizer.BertTokenizer;

import java.util.Arrays;
import java.util.Map;

/**
 */
public class RunChineseRobertaOnOnnx {

    public static void main(String[] args) throws OrtException {
        //Bert-Chinese-Text-Classification-Pytorch项目的 vocab.txt
        String vocabPath = "G:\\qzd\\JavaProject\\QZD_GROUP\\bird-query\\Bert-Chinese-Text-Classification-Pytorch\\chinese_roberta_pretrain\\vocab.txt";
        //bert_to_onnx.py执行后的模型文件
        String modelPath = "G:\\qzd\\JavaProject\\QZD_GROUP\\bird-query\\Bert-Chinese-Text-Classification-Pytorch\\chinese_roberta_pretrain\\saved_dict\\raw_bert_dynamic.onnx";

        String query ="你好，你叫什么名字";

        BertTokenizer bertTokenizer = new BertTokenizer(vocabPath);
        Map<String, OnnxTensor> inputMap = bertTokenizer.tokenizeOnnxTensorForRoberta(Arrays.asList(query));

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
                    float[][][] labelOutput = (float[][][]) resultTensor.getValue();
                    float[][] label = labelOutput[0];
                    float[] label1 = label[0];
                    System.out.println("..."+label1.length);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

}
