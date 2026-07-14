#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace SpongeH5MD
{
namespace detail
{
template <typename T>
inline hid_t Native_H5_Type();

template <>
inline hid_t Native_H5_Type<int>()
{
    return H5T_NATIVE_INT;
}

template <>
inline hid_t Native_H5_Type<float>()
{
    return H5T_NATIVE_FLOAT;
}

inline std::string With_Label(const std::string& data, const std::string& label)
{
    if (label.empty())
    {
        return data;
    }
    return data + " !" + label;
}

template <typename T>
inline std::string Number_String(const T value)
{
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}
}  // namespace detail

class TopologyManybodyH5Materializer
{
   public:
    bool Open(const std::string& file_path)
    {
        last_error_.clear();
        try
        {
            file_.reset(
                new HighFive::File(file_path, HighFive::File::ReadOnly));
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to open topology H5 file: ") +
                        err.what());
        }
    }

    bool Has_EDIP() const { return Exists("/manybody/edip"); }

    bool Has_ReaxFF() const
    {
        return Exists("/manybody/reaxff/parameters") &&
               Exists("/manybody/reaxff/type");
    }

    bool Materialize_EDIP(const std::filesystem::path& path)
    {
        if (!Ensure_File()) return false;
        try
        {
            const int atom_type_count =
                Read_Scalar<int>("/manybody/edip/atom_type_count");
            const auto atom_type =
                Read_Vector<int>("/manybody/edip/atom_type", "EDIP atom type");
            const auto pair_type = Read_Matrix<int>("/manybody/edip/pair/type",
                                                    2, "EDIP pair type");
            const auto pair_parameter = Read_Matrix<float>(
                "/manybody/edip/pair/parameters", 8, "EDIP pair parameters");
            const auto triple_type = Read_Matrix<int>(
                "/manybody/edip/triple/type", 3, "EDIP triple type");
            const auto triple_parameter =
                Read_Matrix<float>("/manybody/edip/triple/parameters", 9,
                                   "EDIP triple parameters");

            if (atom_type_count <= 0)
            {
                return Fail("/manybody/edip/atom_type_count must be positive");
            }
            const std::size_t full_pair_count =
                static_cast<std::size_t>(atom_type_count) *
                static_cast<std::size_t>(atom_type_count);
            if (triple_type.size() / 3 !=
                    full_pair_count *
                        static_cast<std::size_t>(atom_type_count) ||
                triple_parameter.size() / 9 != triple_type.size() / 3)
            {
                return Fail(
                    "/manybody/edip/triple payload does not match "
                    "atom_type_count^3");
            }

            std::map<std::pair<int, int>, std::vector<float>> pair_rows;
            const std::size_t pair_count = pair_type.size() / 2;
            if (pair_parameter.size() / 8 != pair_count)
            {
                return Fail(
                    "/manybody/edip/pair type/parameter row count "
                    "mismatch");
            }
            if (pair_count != full_pair_count &&
                pair_count !=
                    static_cast<std::size_t>(atom_type_count) *
                        static_cast<std::size_t>(atom_type_count + 1) / 2)
            {
                return Fail(
                    "/manybody/edip/pair row count must be either "
                    "atom_type_count^2 or triangular");
            }
            for (std::size_t row = 0; row < pair_count; ++row)
            {
                const int a = pair_type[2 * row];
                const int b = pair_type[2 * row + 1];
                std::vector<float> values(pair_parameter.begin() + 8 * row,
                                          pair_parameter.begin() + 8 * row + 8);
                pair_rows[{a, b}] = values;
                pair_rows[{b, a}] = values;
            }

            std::filesystem::create_directories(path.parent_path());
            std::ofstream out(path);
            if (!out)
            {
                return Fail("failed to create materialized EDIP input: " +
                            path.string());
            }
            out << atom_type.size() << ' ' << atom_type_count << "\n";
            out << "# pair\n";
            for (int a = 0; a < atom_type_count; ++a)
            {
                for (int b = 0; b < atom_type_count; ++b)
                {
                    const auto found = pair_rows.find({a, b});
                    if (found == pair_rows.end())
                    {
                        return Fail("missing EDIP pair parameters for type " +
                                    std::to_string(a) + "," +
                                    std::to_string(b));
                    }
                    out << a << ' ' << b;
                    for (float value : found->second)
                    {
                        out << ' ' << detail::Number_String(value);
                    }
                    out << "\n";
                }
            }
            out << "# triple\n";
            for (std::size_t row = 0; row < triple_type.size() / 3; ++row)
            {
                out << triple_type[3 * row] << ' ' << triple_type[3 * row + 1]
                    << ' ' << triple_type[3 * row + 2];
                for (std::size_t col = 0; col < 9; ++col)
                {
                    out << ' '
                        << detail::Number_String(
                               triple_parameter[9 * row + col]);
                }
                out << "\n";
            }
            out << "# atom types\n";
            for (int value : atom_type)
            {
                out << value << "\n";
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to materialize EDIP from native "
                                    "H5 payload: ") +
                        err.what());
        }
    }

    bool Materialize_ReaxFF(const std::filesystem::path& parameter_path,
                            const std::filesystem::path& type_path)
    {
        if (!Ensure_File()) return false;
        try
        {
            std::filesystem::create_directories(parameter_path.parent_path());
            std::filesystem::create_directories(type_path.parent_path());
            if (!Write_ReaxFF_Parameter_File(parameter_path))
            {
                return false;
            }
            if (!Write_ReaxFF_Type_File(type_path))
            {
                return false;
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to materialize ReaxFF from native "
                                    "H5 payload: ") +
                        err.what());
        }
    }

    std::string Last_Error() const { return last_error_; }

   private:
    bool Ensure_File()
    {
        if (file_ == nullptr)
        {
            return Fail("topology H5 materializer is not open");
        }
        return true;
    }

    bool Exists(const std::string& object_path) const
    {
        return file_ != nullptr && file_->exist(object_path);
    }

    template <typename T>
    T Read_Scalar(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error("dataset is missing: " + dataset_path);
        }
        T value{};
        file_->getDataSet(dataset_path).read(value);
        return value;
    }

    template <typename T>
    std::vector<T> Read_Vector(const std::string& dataset_path,
                               const std::string& label)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 1)
        {
            throw std::runtime_error(label + " dataset " + dataset_path +
                                     " must be one-dimensional");
        }
        std::vector<T> values;
        file_->getDataSet(dataset_path).read(values);
        return values;
    }

    template <typename T>
    std::vector<T> Read_Matrix(const std::string& dataset_path,
                               const std::size_t columns,
                               const std::string& label)
    {
        const auto dims = Dimensions(dataset_path);
        if (dims.size() != 2 || dims[1] != columns)
        {
            std::ostringstream out;
            out << label << " dataset " << dataset_path << " must have shape "
                << "[n," << columns << "]";
            throw std::runtime_error(out.str());
        }
        std::vector<T> values(dims[0] * dims[1]);
        HighFive::DataSet dataset = file_->getDataSet(dataset_path);
        const hsize_t h_dims[2] = {static_cast<hsize_t>(dims[0]),
                                   static_cast<hsize_t>(dims[1])};
        hid_t mem_space = H5Screate_simple(2, h_dims, nullptr);
        if (mem_space < 0)
        {
            throw std::runtime_error(label +
                                     " failed to create memory dataspace at " +
                                     dataset_path);
        }
        const herr_t read_rc =
            H5Dread(dataset.getId(), detail::Native_H5_Type<T>(), mem_space,
                    H5S_ALL, H5P_DEFAULT, values.data());
        H5Sclose(mem_space);
        if (read_rc < 0)
        {
            throw std::runtime_error(label + " failed to read dataset at " +
                                     dataset_path);
        }
        return values;
    }

    std::vector<std::size_t> Dimensions(const std::string& dataset_path)
    {
        if (!Exists(dataset_path))
        {
            throw std::runtime_error("dataset is missing: " + dataset_path);
        }
        return file_->getDataSet(dataset_path).getSpace().getDimensions();
    }

    bool Write_ReaxFF_Type_File(const std::filesystem::path& path)
    {
        const auto names = Read_Vector<std::string>(
            "/manybody/reaxff/type/name", "ReaxFF atom type names");
        const std::int64_t count =
            Read_Scalar<std::int64_t>("/manybody/reaxff/type/count");
        if (count != static_cast<std::int64_t>(names.size()))
        {
            return Fail(
                "/manybody/reaxff/type/count does not match name "
                "length");
        }
        std::ofstream out(path);
        if (!out)
        {
            return Fail("failed to create materialized ReaxFF type input: " +
                        path.string());
        }
        out << names.size() << "\n";
        for (const auto& name : names)
        {
            out << name << "\n";
        }
        return true;
    }

    bool Write_ReaxFF_Parameter_File(const std::filesystem::path& path)
    {
        std::ofstream out(path);
        if (!out)
        {
            return Fail(
                "failed to create materialized ReaxFF parameter input: " +
                path.string());
        }

        const std::string root = "/manybody/reaxff/parameters";
        out << Read_Scalar<std::string>(root + "/header") << "\n";
        if (!Write_ReaxFF_General(out, root + "/general")) return false;
        if (!Write_ReaxFF_Atom(out, root + "/atom")) return false;
        if (!Write_ReaxFF_Bond(out, root + "/bond")) return false;
        if (!Write_ReaxFF_Simple_Count_Section(out, root + "/off_diagonal", 2,
                                               6))
        {
            return false;
        }
        if (!Write_ReaxFF_Simple_Count_Section(out, root + "/angle", 3, 7))
        {
            return false;
        }
        if (!Write_ReaxFF_Simple_Count_Section(out, root + "/torsion", 4, 7))
        {
            return false;
        }
        if (!Write_ReaxFF_Simple_Count_Section(out, root + "/hydrogen_bond", 3,
                                               4))
        {
            return false;
        }
        return true;
    }

    bool Write_ReaxFF_General(std::ofstream& out, const std::string& root)
    {
        const auto values =
            Read_Vector<float>(root + "/value", "ReaxFF general values");
        const auto labels =
            Read_Vector<std::string>(root + "/label", "ReaxFF general labels");
        const auto count = Read_Scalar<std::int64_t>(root + "/count");
        const auto count_label =
            Read_Scalar<std::string>(root + "/count_label");
        if (count != static_cast<std::int64_t>(values.size()) ||
            values.size() != labels.size())
        {
            return Fail("ReaxFF general count/value/label length mismatch");
        }
        if (count <= 38)
        {
            return Fail(
                "ReaxFF native H5 general parameter payload is "
                "incomplete for runtime initialization");
        }
        out << detail::With_Label(std::to_string(count), count_label) << "\n";
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            out << detail::With_Label(detail::Number_String(values[i]),
                                      labels[i])
                << "\n";
        }
        return true;
    }

    bool Write_ReaxFF_Atom(std::ofstream& out, const std::string& root)
    {
        const auto count = Read_Scalar<std::int64_t>(root + "/count");
        const auto count_label =
            Read_Scalar<std::string>(root + "/count_label");
        const auto headers =
            Read_Vector<std::string>(root + "/header", "ReaxFF atom header");
        const auto type_names = Read_Vector<std::string>(
            root + "/type_name", "ReaxFF atom type name");
        const auto values =
            Read_Vector<float>(root + "/value", "ReaxFF atom values");
        const auto line_labels = Read_Vector<std::string>(
            root + "/line_label", "ReaxFF atom line labels");
        const auto line_offsets = Read_Vector<std::int64_t>(
            root + "/line_value_offset", "ReaxFF atom line value offsets");
        if (headers.size() != 3 ||
            type_names.size() != static_cast<std::size_t>(count) ||
            line_labels.size() != static_cast<std::size_t>(4 * count) ||
            line_offsets.size() != line_labels.size() + 1 ||
            line_offsets.back() != static_cast<std::int64_t>(values.size()))
        {
            return Fail("ReaxFF atom section shape is invalid");
        }
        out << detail::With_Label(std::to_string(count), count_label) << "\n";
        for (const auto& header : headers)
        {
            out << header << "\n";
        }
        for (std::size_t atom = 0; atom < type_names.size(); ++atom)
        {
            for (std::size_t local_line = 0; local_line < 4; ++local_line)
            {
                const std::size_t line = atom * 4 + local_line;
                std::ostringstream row;
                if (local_line == 0)
                {
                    row << type_names[atom];
                }
                const auto begin = static_cast<std::size_t>(line_offsets[line]);
                const auto end =
                    static_cast<std::size_t>(line_offsets[line + 1]);
                for (std::size_t value = begin; value < end; ++value)
                {
                    if (row.tellp() > 0) row << ' ';
                    row << detail::Number_String(values[value]);
                }
                out << detail::With_Label(row.str(), line_labels[line]) << "\n";
            }
        }
        return true;
    }

    bool Write_ReaxFF_Bond(std::ofstream& out, const std::string& root)
    {
        const auto count = Read_Scalar<std::int64_t>(root + "/count");
        const auto count_label =
            Read_Scalar<std::string>(root + "/count_label");
        const auto headers =
            Read_Vector<std::string>(root + "/header", "ReaxFF bond header");
        const auto types =
            Read_Matrix<int>(root + "/type", 2, "ReaxFF bond type");
        const auto values =
            Read_Vector<float>(root + "/value", "ReaxFF bond values");
        const auto line_labels = Read_Vector<std::string>(
            root + "/line_label", "ReaxFF bond line labels");
        const auto line_offsets = Read_Vector<std::int64_t>(
            root + "/line_value_offset", "ReaxFF bond line value offsets");
        if (headers.size() != 1 ||
            types.size() / 2 != static_cast<std::size_t>(count) ||
            line_labels.size() != static_cast<std::size_t>(2 * count) ||
            line_offsets.size() != line_labels.size() + 1 ||
            line_offsets.back() != static_cast<std::int64_t>(values.size()))
        {
            return Fail("ReaxFF bond section shape is invalid");
        }
        out << detail::With_Label(std::to_string(count), count_label) << "\n";
        out << headers[0] << "\n";
        for (std::size_t bond = 0; bond < static_cast<std::size_t>(count);
             ++bond)
        {
            for (std::size_t local_line = 0; local_line < 2; ++local_line)
            {
                const std::size_t line = bond * 2 + local_line;
                std::ostringstream row;
                if (local_line == 0)
                {
                    row << types[2 * bond] << ' ' << types[2 * bond + 1];
                }
                const auto begin = static_cast<std::size_t>(line_offsets[line]);
                const auto end =
                    static_cast<std::size_t>(line_offsets[line + 1]);
                for (std::size_t value = begin; value < end; ++value)
                {
                    if (row.tellp() > 0) row << ' ';
                    row << detail::Number_String(values[value]);
                }
                out << detail::With_Label(row.str(), line_labels[line]) << "\n";
            }
        }
        return true;
    }

    bool Write_ReaxFF_Simple_Count_Section(std::ofstream& out,
                                           const std::string& root,
                                           const std::size_t type_columns,
                                           const std::size_t value_columns)
    {
        const auto count = Read_Scalar<std::int64_t>(root + "/count");
        const auto count_label =
            Read_Scalar<std::string>(root + "/count_label");
        const auto types = Read_Matrix<int>(root + "/type", type_columns,
                                            "ReaxFF counted section type");
        const auto values = Read_Matrix<float>(root + "/value", value_columns,
                                               "ReaxFF counted section values");
        if (types.size() / type_columns != static_cast<std::size_t>(count) ||
            values.size() / value_columns != static_cast<std::size_t>(count))
        {
            return Fail("ReaxFF counted section row count mismatch at " + root);
        }
        out << detail::With_Label(std::to_string(count), count_label) << "\n";
        for (std::size_t row = 0; row < static_cast<std::size_t>(count); ++row)
        {
            for (std::size_t col = 0; col < type_columns; ++col)
            {
                if (col > 0) out << ' ';
                out << types[type_columns * row + col];
            }
            for (std::size_t col = 0; col < value_columns; ++col)
            {
                out << ' '
                    << detail::Number_String(values[value_columns * row + col]);
            }
            out << "\n";
        }
        return true;
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        return false;
    }

    std::unique_ptr<HighFive::File> file_;
    std::string last_error_;
};

template <typename ControllerType>
inline bool Materialize_Native_Manybody_Text_Inputs_From_H5(
    ControllerType* controller, const std::string& topology_h5_path,
    const std::filesystem::path& output_dir, std::string* error_message)
{
    auto fail = [error_message](const std::string& message)
    {
        if (error_message != nullptr)
        {
            *error_message = message;
        }
        return false;
    };

    if (controller == nullptr)
    {
        return fail("controller pointer is null");
    }

    TopologyManybodyH5Materializer materializer;
    if (!materializer.Open(topology_h5_path))
    {
        return fail(materializer.Last_Error());
    }

    if (materializer.Has_EDIP() &&
        !controller->Command_Exist("EDIP", "in_file"))
    {
        const auto edip_path =
            std::filesystem::absolute(output_dir / "edip.txt")
                .lexically_normal();
        if (!materializer.Materialize_EDIP(edip_path))
        {
            return fail(materializer.Last_Error());
        }
        controller->Set_Command("EDIP_in_file", edip_path.string().c_str(), 0);
    }

    const bool has_reaxff_in_file =
        controller->Command_Exist("REAXFF", "in_file");
    const bool has_reaxff_type_in_file =
        controller->Command_Exist("REAXFF", "type_in_file");
    if (materializer.Has_ReaxFF() && !has_reaxff_in_file &&
        !has_reaxff_type_in_file)
    {
        const auto parameter_path =
            std::filesystem::absolute(output_dir / "reaxff.txt")
                .lexically_normal();
        const auto type_path =
            std::filesystem::absolute(output_dir / "reaxff_type.txt")
                .lexically_normal();
        if (!materializer.Materialize_ReaxFF(parameter_path, type_path))
        {
            return fail(materializer.Last_Error());
        }
        controller->Set_Command("REAXFF_in_file",
                                parameter_path.string().c_str(), 0);
        controller->Set_Command("REAXFF_type_in_file",
                                type_path.string().c_str(), 0);
    }
    else if (materializer.Has_ReaxFF() &&
             has_reaxff_in_file != has_reaxff_type_in_file)
    {
        return fail(
            "partial ReaxFF legacy override is invalid: both "
            "REAXFF_in_file and REAXFF_type_in_file are required");
    }
    return true;
}
}  // namespace SpongeH5MD
