package org.example.onnx;


import ai.onnxruntime.*;
import cn.hutool.core.collection.ListUtil;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.example.onnx.tokenizer.BertTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 *
 */
public class RunUbertaOnOnnx {

    public static void main(String[] args) {
        try {
            String query = "雷鸣山";
            List<JSONObject> predict = predict(query);
            System.out.println(JSON.toJSONString(predict));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static List<JSONObject> predict(String query) throws Exception {
        //Bert-Chinese-Text-Classification-Pytorch项目的 vocab.txt
        String vocabPath = "/data/modelfiles/eric/ubert_pretrain/vocab.txt";
        //bert_to_onnx.py执行后的模型文件
        String modelPath = "/data/modelfiles/eric/ner_opti_12_14_v4.onnx";

//        String query = "雷鸣山企知道";

        List<JSONObject> jsonObjects = composeQueries(query);

        BertTokenizer bertTokenizer = new BertTokenizer(vocabPath);
        Map<String, OnnxTensor> inputMap = bertTokenizer.encode(jsonObjects);

        OrtEnvironment env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        /*GPU start----*/
//        OrtCUDAProviderOptions cudaOpts = new OrtCUDAProviderOptions(0);
//        cudaOpts.add("gpu_mem_limit", "" + (512 * 1024 * 1024));
//        options.addCUDA(cudaOpts);
//        int gpuDeviceId = 0;
//        options.addCUDA(gpuDeviceId);
        /*GPU end----*/
        try (OrtSession session = env.createSession(modelPath, options)) {
            // Load code not shown for brevity.
            // Run the inference
            try (OrtSession.Result results = session.run(inputMap)) {
                // Only iterates once
                for (Map.Entry<String, OnnxValue> r : results) {
                    OnnxValue resultValue = r.getValue();
                    OnnxTensor resultTensor = (OnnxTensor) resultValue;

                    double[][][][] labelOutput = (double[][][][]) resultTensor.getValue();
                    long[][] pos = new long[2][];
                    int index = 0;
                    for (int i = 0; i < labelOutput.length; i++) {
                        double[][][] lable1 = labelOutput[i];
                        for (int j = 0; j < lable1.length; j++) {
                            double[][] lable2 = lable1[j];
                            for (int k = 0; k < lable2.length; k++) {
                                double[] label3 = lable2[k];
                                for (int l = 0; l < label3.length; l++) {
                                    double x = label3[l];
                                    double nx = sigmoid(x);
                                    label3[l] = nx;
                                    if (nx > 0.5D) {
                                        System.out.println(i + "," + j + "," + k + "," + l);
                                        long[] p = {i, j, k, l};
                                        pos[index++] = p;
                                    }
                                }
                            }
                        }
                    }

                    bertTokenizer.decode(labelOutput,pos,jsonObjects);
                    List<JSONObject> entityList = extract_entities(jsonObjects);
                    return entityList;
//                    System.out.println(JSON.toJSONString(entityList));
//                    double[][][] label = labelOutput[0];
//                    double[][] label1 = label[0];
//                    System.out.println("..." + label1.length);
                }


            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    private static List<JSONObject> extract_entities(List<JSONObject> jsonObjects) {
        List<JSONObject> res_list = new ArrayList<>();

        for (int i = 0; i < jsonObjects.size(); i++) {
            JSONObject inp = jsonObjects.get(i);
            String query = inp.getString("text");

            JSONObject res = new JSONObject();
            res.put("query",query);

            JSONArray choices = inp.getJSONArray("choices");
            for (int j = 0; j < choices.size(); j++) {
                JSONObject ent = choices.getJSONObject(j);
                String k = ent.getString("entity_type");
                JSONArray v_full_list = ent.getJSONArray("entity_list");

                JSONArray v_list = new JSONArray();
                v_list.addAll(v_full_list);
                res.put(k,v_list);
            }
            res_list.add(res);

        }

        return res_list;
    }


    public static double sigmoid(double x) {
        x = Math.max(-500, x);
        x = Math.min(500, x);
        return 1 / (1 + Math.exp(-x));
    }

    private static List<JSONObject> composeQueries(String query) {
        JSONObject requestBody = new JSONObject();
        requestBody.put("id", 0);
        requestBody.put("subtask_type", "实体识别");
        requestBody.put("task_type", "抽取任务");
        requestBody.put("text", query);

        JSONArray choices = new JSONArray();
        List<String> categorys = ListUtil.of("人名", "地名", "公司", "行业", "公司类别", "品牌");
        for (String category : categorys) {
            JSONObject cat = new JSONObject();
            cat.put("entity_type", category);
            choices.add(cat);
        }

        requestBody.put("choices", choices);
        return ListUtil.of(requestBody);
    }

}
