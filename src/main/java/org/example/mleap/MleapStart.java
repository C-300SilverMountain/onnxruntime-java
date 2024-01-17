package org.example.mleap;

import cn.hutool.core.io.FileUtil;
import cn.hutool.json.JSONUtil;
import ml.combust.mleap.core.types.StructField;
import ml.combust.mleap.core.types.StructType;
import ml.combust.mleap.runtime.MleapContext;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import ml.combust.mleap.runtime.frame.Row;
import ml.combust.mleap.runtime.frame.Transformer;
import ml.combust.mleap.runtime.javadsl.BundleBuilder;
import ml.combust.mleap.runtime.javadsl.ContextBuilder;
import ml.combust.mleap.runtime.javadsl.LeapFrameBuilder;
import ml.combust.mleap.tensor.Tensor;
import scala.collection.Iterator;
import scala.collection.Seq;

import java.util.ArrayList;
import java.util.List;

/**
 * 开源AI模型序列化总结: https://www.jianshu.com/p/777d07037d00
 */

/**
 * https://github.com/combust/mleap
 * https://stackoverflow.com/questions/44446133/scala-to-java-8-mleap-translation
 * MLeap 中文文档: https://www.bookstack.cn/read/mleap-zh/notebooks-index.md
 */

/**
 * MLeap 中文文档: https://www.bookstack.cn/read/mleap-zh/notebooks-index.md
 */

/**
 * Elasticsearch 整合机器学习强化排序 : https://segmentfault.com/a/1190000043650417
 * OpenVINO简介: https://zhuanlan.zhihu.com/p/638671049
 * OpenVINO官网：https://docs.openvino.ai/2023.2/ovms_docs_deploying_server.html
 */

/**
 * 第一步：执行 mlean-start.py 生成 mleap-scikit-test-pipeline.zip ，源码：https://github.com/combust/mleap 之【Create and Export a Scikit-Learn Pipeline】
 * 第二步：执行当前类，运行结果：
 * [1,0,0]
 * [0,1,0]
 * [0,0,1]
 */
public class MleapStart {

    private static Transformer kMeansModel;
    private static MleapContext mleapContext;
    private static BundleBuilder bundleBuilder;

    static {
        mleapContext = new ContextBuilder().createMleapContext();
        bundleBuilder = new BundleBuilder();
//        Resource res = ResourceLoader.getResource("classpath:aihello.com/aimodels/kmeans-model.zip");
        kMeansModel = bundleBuilder.load(FileUtil.file("G:\\qzd\\JavaProject\\QZD_GROUP\\bird-query\\mleap-py\\mleap-scikit-test-pipeline.zip"), mleapContext).root();
    }


    public static void main(String[] args) {

        LeapFrameBuilder builder = new LeapFrameBuilder();
        List<StructField> fields = new ArrayList<StructField>();
        fields.add(builder.createField("col_a", builder.createString()));
        StructType schema = builder.createSchema(fields);

        List<Row> rows = new ArrayList<Row>();
        rows.add(builder.createRow("a"));
        rows.add(builder.createRow("b"));
        rows.add(builder.createRow("c"));

        DefaultLeapFrame frame = builder.createFrame(schema, rows);
        DefaultLeapFrame returnFrame = kMeansModel.transform(frame).get();
        Seq<Row> dataset = returnFrame.dataset();
        Iterator<Row> iterator = dataset.iterator();
        while (iterator.hasNext()) {
            Row next = iterator.next();
            Tensor<Object> tensor = next.getTensor(2);
            System.out.println(JSONUtil.toJsonStr(tensor.toArray()));
        }

    }
}
