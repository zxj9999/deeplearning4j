/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/lrn.h>

namespace nd4j {
namespace ops {
namespace helpers {

#ifdef HAVE_MKLDNN
using namespace mkldnn;

template <typename T>
static void getMKLDNNMemoryDescLrn(const NDArray<T>* src, const NDArray<T>* diff_src,
        mkldnn::memory::desc* lrn_src_md, mkldnn::memory::desc* lrn_diff_src_md, int axis) {
    const Nd4jLong* shape = src->getShapeInfo();
    long rank = shape[0];
    long dim1 = axis; // MKL-DNN supports only 1 axis, which has to be the "channel" one
    long dim2 = axis >= 2 ? 1 : 2;
    long dim3 = axis >= 3 ? 2 : 3;
    mkldnn::memory::dims lrn_src_tz = { (int)shape[1], (int)shape[dim1 + 1], rank > 2 ? (int)shape[dim2 + 1] : 1, rank > 3 ? (int)shape[dim3 + 1] : 1};

    auto type = mkldnn::memory::data_type::f32;
    auto format = axis == 1 ? mkldnn::memory::format::nchw : mkldnn::memory::format::nhwc;

    if (src != nullptr && src->getBuffer() != nullptr && lrn_src_md != nullptr) {
        *lrn_src_md = mkldnn::memory::desc({ lrn_src_tz }, type, format);
        // lrn_src_md->data.format = mkldnn_blocked; // unsupported for lrn, leave nchw or nhwc for now
        lrn_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[0];
        lrn_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[dim1];
        lrn_src_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? src->stridesOf()[dim2] : 1;
        lrn_src_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? src->stridesOf()[dim3] : 1;
    }

    if (diff_src != nullptr && diff_src->getBuffer() != nullptr && lrn_diff_src_md != nullptr) {
        *lrn_diff_src_md = mkldnn::memory::desc({ lrn_src_tz }, type, format);
        // lrn_diff_src_md->data.format = mkldnn_blocked; // unsupported for lrn, leave nchw or nhwc for now
        lrn_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[0];
        lrn_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[dim1];
        lrn_diff_src_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? diff_src->stridesOf()[dim2] : 1;
        lrn_diff_src_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? diff_src->stridesOf()[dim3] : 1;
    }
}
#endif

    template <typename T>
    int lrnFunctor(nd4j::graph::Context<T>& block, NDArray<T>* input, NDArray<T>* output, int depth, T bias, T alpha, T beta) {

        T dividor;

        int totalLength = input->lengthOf();
        int lastDim = input->sizeAt(-1);
        int chunkCount = totalLength / lastDim;

#ifdef HAVE_MKLDNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream<T>::isSupported()) {
        std::vector<nd4j::MKLDNNStream<T> >& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream<T>("lrn"));
        }

        if (streams[0].checkAndReset({input}, {output}, {bias, alpha, beta}, {depth})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc lrn_src_md(empty);

            getMKLDNNMemoryDescLrn<T>(input, nullptr, &lrn_src_md, nullptr, input->rankOf() - 1);

            auto lrn_desc = lrn_forward::desc(prop_kind::forward_inference, lrn_across_channels, lrn_src_md, (2 * depth + 1), alpha * (2 * depth + 1), beta, bias);

            auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, streams[0].getEngine());
            auto lrn_src_memory = mkldnn::memory(lrn_prim_desc.src_primitive_desc(), input->buffer());
            auto lrn_dst_memory = mkldnn::memory(lrn_prim_desc.dst_primitive_desc(), output->buffer());
            streams[0].setMemory({lrn_src_memory, lrn_dst_memory});
            streams[0].setOperation(lrn_forward(lrn_prim_desc, lrn_src_memory, lrn_dst_memory));
        }

        streams[0].submitAndWait();
        return ND4J_STATUS_OK;
    }
#endif
    nd4j_debug("MKL-DNN is not used for lrn!\n", 0);

        std::unique_ptr<ResultSet<T>> listOut(output->allTensorsAlongDimension({output->rankOf() - 1}));
        std::unique_ptr<ResultSet<T>> listInput(input->allTensorsAlongDimension({input->rankOf() - 1}));
        if (chunkCount != listOut->size()) 
            return ND4J_STATUS_VALIDATION;
        for (int c = 0; c < chunkCount; c++) {
            for (int e = 0; e < lastDim; e++) {
                int begin = nd4j::math::nd4j_max(0, e - depth);
                int end = nd4j::math::nd4j_min(depth + e + 1, lastDim);
                T quadSum = 0;

                for (int pos = begin; pos < end; ++pos) {
                    T val = (*listInput->at(c))(pos);
                    quadSum += val * val;
                }
                T dividor = nd4j::math::nd4j_pow(bias + alpha * quadSum, beta);
                (*listOut->at(c))(e) = (*listInput->at(c))(e) / dividor;
            }
        }

        return ND4J_STATUS_OK;
    }

    template <typename T>
    int lrnFunctorEx(nd4j::graph::Context<T>& block, NDArray<T>* input, NDArray<T>* output, NDArray<T>* unitScale, NDArray<T>* scale, int depth, T bias, T alpha, T beta) {
    
        depth = nd4j::math::nd4j_min<Nd4jLong>(depth, input->sizeAt(1));

        int halfDepth = (int) ( (T) depth / (T) 2.f);
        halfDepth = nd4j::math::nd4j_max(halfDepth, 0);
        const int channel =  input->sizeAt(1);

#if 0
//#ifdef HAVE_MKLDNN
//XXX: need to get output to match exactly with MKL-DNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream<T>::isSupported()) {
        std::vector<nd4j::MKLDNNStream<T> >& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream<T>("lrn_bp"));
        }

        if (streams[0].checkAndReset({input, scale}, {output}, {bias, alpha, beta}, {depth})) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc lrn_src_md(empty), lrn_diff_src_md(empty);

            getMKLDNNMemoryDescLrn<T>(input, scale, &lrn_src_md, &lrn_diff_src_md, 1);

            auto lrn_desc = lrn_forward::desc(prop_kind::forward, lrn_across_channels, lrn_src_md, (2 * halfDepth + 1), alpha * (2 * halfDepth + 1), beta, bias);
            auto lrn_back_desc = lrn_backward::desc(lrn_across_channels, lrn_src_md, lrn_diff_src_md, (2 * halfDepth + 1), alpha * (2 * halfDepth + 1), beta, bias);

            auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, streams[0].getEngine());
            auto lrn_back_prim_desc = lrn_backward::primitive_desc(lrn_back_desc, streams[0].getEngine(), lrn_prim_desc);
            auto lrn_src_memory = mkldnn::memory(lrn_prim_desc.src_primitive_desc(), input->buffer());
            auto lrn_dst_memory = mkldnn::memory(lrn_back_prim_desc.diff_dst_primitive_desc(), scale->buffer());
            auto lrn_diff_src_memory = mkldnn::memory(lrn_back_prim_desc.diff_src_primitive_desc(), output->buffer());
            streams[0].setMemory({lrn_src_memory, lrn_dst_memory, lrn_diff_src_memory});
            streams[0].setOperation(lrn_backward(lrn_back_prim_desc, lrn_src_memory, lrn_dst_memory, lrn_diff_src_memory));
        }

        streams[0].submitAndWait();
        return ND4J_STATUS_OK;
    }
#endif
    nd4j_debug("MKL-DNN is not used for lrn_bp!\n", 0);

        std::unique_ptr<NDArray<T>> activitySqr(input->dup('c'));//NDArrayFactory<T>::createUninitialized(input));
        std::unique_ptr<NDArray<T>> sumPart(activitySqr->dup('c'));

        input->template applyPairwiseTransform<simdOps::Multiply<T>>(input, activitySqr.get(), nullptr);
#pragma omp parallel for if (halfDepth + 1 > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
        for (int i = 1; i < halfDepth + 1; i++) {
            IndicesList indA({NDIndex::all(), NDIndex::interval(i, channel), NDIndex::all(), NDIndex::all()});
            IndicesList indB({NDIndex::all(), NDIndex::interval(0, channel - i), NDIndex::all(), NDIndex::all()});

            std::unique_ptr<NDArray<T>> tmp(sumPart->subarray(indA));
            std::unique_ptr<NDArray<T>> addVal(activitySqr->subarray(indB));

            tmp->template applyPairwiseTransform<simdOps::Add<T>>(addVal.get(), nullptr);


            std::unique_ptr<NDArray<T>> tmp2(sumPart->subarray(indB));
            std::unique_ptr<NDArray<T>> addVal2(activitySqr->subarray(indA));

            tmp2->template applyPairwiseTransform<simdOps::Add<T>>(addVal2.get(), nullptr);
        }

        /*
         *  // taken from java
            unitScale = sumPart.mul(alpha).addi(k).leverageTo(ComputationGraph.workspaceExternal);
            // y = x * unitScale**-beta
            scale = Transforms.pow(unitScale, -beta).leverageTo(ComputationGraph.workspaceExternal);
            activations = input.mul(scale).leverageTo(ComputationGraph.workspaceExternal);
         */
        if (unitScale != nullptr && scale != nullptr) {
            sumPart->template applyScalar<simdOps::Multiply<T>>(alpha, unitScale, nullptr);
            unitScale->template applyScalar<simdOps::Add<T>>(bias);

            T p = -beta;
            unitScale->template applyTransform<simdOps::Pow<T>>(scale, &p);
            input->template applyPairwiseTransform<simdOps::Multiply<T>>(scale, output, nullptr);
        }

        return ND4J_STATUS_OK;
    }

    template int lrnFunctor(nd4j::graph::Context<float>& block, NDArray<float>* input, NDArray<float>* output, int depth, float bias, float alpha, float beta);
    template int lrnFunctor(nd4j::graph::Context<float16>& block, NDArray<float16>* input, NDArray<float16>* output, int depth, float16 bias, float16 alpha, float16 beta);
    template int lrnFunctor(nd4j::graph::Context<double>& block, NDArray<double>* input, NDArray<double>* output, int depth, double bias, double alpha, double beta);
    template int lrnFunctorEx(nd4j::graph::Context<float>& block, NDArray<float>* input, NDArray<float>* output, NDArray<float>* unitScale, NDArray<float>* scale, int depth, float bias, float alpha, float beta);
    template int lrnFunctorEx(nd4j::graph::Context<float16>& block, NDArray<float16>* input, NDArray<float16>* output, NDArray<float16>* unitScale, NDArray<float16>* scale, int depth, float16 bias, float16 alpha, float16 beta);
    template int lrnFunctorEx(nd4j::graph::Context<double>& block, NDArray<double>* input, NDArray<double>* output, NDArray<double>* unitScale, NDArray<double>* scale, int depth, double bias, double alpha, double beta);
}
}
}
