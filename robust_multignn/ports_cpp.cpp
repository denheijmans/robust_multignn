#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace py = pybind11;

struct Triplet {
    int64_t first;   // neighbor node id
    double second;   // timestamp (for sort order)
    int64_t third;   // original edge index
    Triplet(int64_t f, double s, int64_t t) : first(f), second(s), third(t) {}
};

typedef std::unordered_map<int64_t, std::vector<Triplet>> AdjList;

py::array_t<int64_t> ports(py::ssize_t M, const AdjList& adj_list) {
    py::array_t<int64_t> ports_array(M);
    auto ports_r = ports_array.mutable_unchecked<1>();
    std::vector<Triplet> sorted_nbs;

    for (const auto& [v, nbs] : adj_list) {
        if (nbs.empty()) continue;
        std::unordered_map<int64_t, int64_t> mapping;
        int64_t counter = 0;
        sorted_nbs = nbs;

        std::sort(sorted_nbs.begin(), sorted_nbs.end(), [](const Triplet& a, const Triplet& b) {
            return a.second < b.second;
        });

        for (const auto& [target, timestamp, idx] : sorted_nbs) {
            auto it = mapping.find(target);
            if (it != mapping.end()) {
                ports_r(idx) = it->second;
            } else {
                mapping[target] = counter;
                ports_r(idx) = counter;
                counter++;
            }
        }
    }
    return ports_array;
}

py::tuple assign_ports(
    py::array_t<int64_t> edge_src,
    py::array_t<int64_t> edge_dst,
    py::array_t<double> time,
    int64_t num_nodes
) {
    auto src_r = edge_src.unchecked<1>();
    auto dst_r = edge_dst.unchecked<1>();
    auto time_r = time.unchecked<1>();
    py::ssize_t M = edge_src.shape(0);

    AdjList adj_list_out;
    AdjList adj_list_in;
    adj_list_out.reserve(num_nodes);
    adj_list_in.reserve(num_nodes);

    for (py::ssize_t i = 0; i < M; ++i) {
        int64_t u = src_r(i);
        int64_t v = dst_r(i);
        double t = time_r(i);
        adj_list_out[u].emplace_back(v, t, i);
        adj_list_in[v].emplace_back(u, t, i);
    }

    py::array_t<int64_t> ports_in = ports(M, adj_list_in);
    py::array_t<int64_t> ports_out = ports(M, adj_list_out);

    return py::make_tuple(ports_in, ports_out);
}

PYBIND11_MODULE(robust_multignn_ports_cpp, m) {
    m.doc() = "Fast C++ port assignment for graph edges";
    m.def("assign_ports", &assign_ports,
        py::arg("edge_src"), py::arg("edge_dst"), py::arg("time"), py::arg("num_nodes"));
}
