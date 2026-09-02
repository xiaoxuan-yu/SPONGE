#pragma once

void MD_INFORMATION::periodic_box_condition_information::Initial(
    CONTROLLER* controller, MD_INFORMATION* md_info)
{
    this->md_info = md_info;
    this->pbc = true;
    if (controller->Command_Exist("pbc"))
    {
        this->pbc = controller->Get_Bool(
            "pbc",
            "MD_INFORMATION::periodic_box_condition_information::Initial");
    }
    this->No_PBC_Check(controller);
    this->PBC_Check();
    this->cell0 = cell;
}

void MD_INFORMATION::periodic_box_condition_information::No_PBC_Check(
    CONTROLLER* controller)
{
    if (this->pbc) return;

    if (controller->MPI_size > 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "NOPBC can not be used in Multi-Process mode");
    }

    if (md_info->nb.cutoff < 100)
    {
        controller->Warn(
            "The cutoff for NOPBC is not greater than 100 angstrom, which may "
            "be inaccurate");
    }
    if (md_info->sys.box_length.x < 900 || md_info->sys.box_length.y < 900 ||
        md_info->sys.box_length.z < 900)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "The box length of the system should always be greater than 900 "
            "angstrom for NOPBC");
    }
    if (md_info->mode == md_info->NPT)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "NPT mode can not be used for NOPBC");
    }
    if (!(controller->Command_Exist("SITS", "atom_numbers") &&
          (strcmp(controller->Command("SITS", "atom_numbers"), "ITS") == 0 ||
           strcmp(controller->Command("SITS", "atom_numbers"), "ALL") == 0)) &&
        controller->Command_Exist("SITS", "mode"))
    {
        controller->Throw_SPONGE_Error(
            spongeErrorConflictingCommand,
            "MD_INFORMATION::periodic_box_condition_information::No_PBC_Check",
            "SITS can not be used for NOPBC now");
    }
}

void MD_INFORMATION::periodic_box_condition_information::PBC_Check()
{
    VECTOR length = md_info->sys.box_length;
    VECTOR angle = CONSTANT_DEG_TO_RAD * md_info->sys.box_angle;
    float a = length.x;
    float b = length.y;
    float c = length.z;
    float alpha = angle.x;
    float beta = angle.y;
    float gamma = angle.z;
    float za = sqrtf(1 - cosf(alpha) * cosf(alpha) - cosf(beta) * cosf(beta) -
                     cosf(gamma) * cosf(gamma) +
                     2 * cosf(alpha) * cosf(beta) * cosf(gamma));

    cell.a11 = a;
    cell.a21 = b * cosf(gamma);
    cell.a22 = b * sinf(gamma);
    cell.a31 = c * cosf(beta);
    cell.a32 = c / sinf(gamma) * (cosf(alpha) - cosf(beta) * cosf(gamma));
    cell.a33 = c / sinf(gamma) * za;

    rcell.a11 = 1.0f / cell.a11;
    rcell.a22 = 1.0f / cell.a22;
    rcell.a33 = 1.0f / cell.a33;
    rcell.a21 = -rcell.a11 / tanf(gamma);
    rcell.a31 = (cosf(alpha) / tanf(gamma) - cosf(beta) / sinf(gamma)) / za / a;
    rcell.a32 = (cosf(beta) / tanf(gamma) - cosf(alpha) / sinf(gamma)) / za / b;

    cell.a21 = fabsf(cell.a21) < 1e-3 ? 0 : cell.a21;
    cell.a31 = fabsf(cell.a31) < 1e-3 ? 0 : cell.a31;
    cell.a32 = fabsf(cell.a32) < 1e-3 ? 0 : cell.a32;

    rcell.a21 = fabsf(rcell.a21) < 1e-3 ? 0 : rcell.a21;
    rcell.a31 = fabsf(rcell.a31) < 1e-3 ? 0 : rcell.a31;
    rcell.a32 = fabsf(rcell.a32) < 1e-3 ? 0 : rcell.a32;
}

void MD_INFORMATION::periodic_box_condition_information::Update_Box(LTMatrix3 g)
{
    cell.a11 = cell.a11 + md_info->dt * cell.a11 * g.a11;
    cell.a22 = cell.a22 + md_info->dt * cell.a22 * g.a22;
    cell.a33 = cell.a33 + md_info->dt * cell.a33 * g.a33;
    cell.a21 = cell.a21 + md_info->dt * (cell.a21 * g.a11 + cell.a22 * g.a21);
    cell.a31 = cell.a31 + md_info->dt * (cell.a31 * g.a11 + cell.a32 * g.a21 +
                                         cell.a33 * g.a31);
    cell.a32 = cell.a32 + md_info->dt * (cell.a32 * g.a22 + cell.a33 * g.a32);
    VECTOR va = {cell.a11, 0, 0};
    VECTOR vb = {cell.a21, cell.a22, 0};
    VECTOR vc = {cell.a31, cell.a32, cell.a33};
    float a = sqrtf(va * va);
    float b = sqrtf(vb * vb);
    float c = sqrtf(vc * vc);
    float alpha = acos(va * vb / a / b);
    float beta = acos(va * vc / a / c);
    float gamma = acos(vc * vb / c / b);
    float za = sqrtf(1 - cosf(alpha) * cosf(alpha) - cosf(beta) * cosf(beta) -
                     cosf(gamma) * cosf(gamma) +
                     2 * cosf(alpha) * cosf(beta) * cosf(gamma));
    rcell.a11 = 1.0f / cell.a11;
    rcell.a22 = 1.0f / cell.a22;
    rcell.a33 = 1.0f / cell.a33;
    rcell.a21 = -rcell.a11 / tanf(gamma);
    rcell.a31 = (cosf(alpha) / tanf(gamma) - cosf(beta) / sinf(gamma)) / za / a;
    rcell.a32 = (cosf(beta) / tanf(gamma) - cosf(alpha) / sinf(gamma)) / za / b;

    md_info->sys.box_length.x = a;
    md_info->sys.box_length.y = b;
    md_info->sys.box_length.z = c;
    md_info->sys.box_angle.x = alpha * CONSTANT_RAD_TO_DEG;
    md_info->sys.box_angle.y = beta * CONSTANT_RAD_TO_DEG;
    md_info->sys.box_angle.z = gamma * CONSTANT_RAD_TO_DEG;
}

bool MD_INFORMATION::periodic_box_condition_information::Check_Change_Large(
    float neighbor_skin)
{
    bool result = false;
    float grid_length = 0.5f * (md_info->nb.cutoff + neighbor_skin);
    float* cell = (float*)&this->cell;
    float* cell0 = (float*)&this->cell0;
    int i1, i0;
    float f1, f0;
    for (int i = 0; i < 6; i += 1)
    {
        i1 = cell[i] / grid_length;
        i0 = cell0[i] / grid_length;
        f1 = cell[i];
        f0 = cell0[i];
        if (fabsf(f1 - f0) > 0.5f * neighbor_skin && i1 != i0)
        {
            result = true;
        }
    }
    if (result)
    {
        this->cell0 = this->cell;
    }
    return result;
}

LTMatrix3 MD_INFORMATION::periodic_box_condition_information::Get_Cell(
    VECTOR box_length, VECTOR box_angle)
{
    LTMatrix3 cell;
    double a = box_length.x;
    double b = box_length.y;
    double c = box_length.z;
    double alpha = CONSTANT_DEG_TO_RAD_DOUBLE * box_angle.x;
    double beta = CONSTANT_DEG_TO_RAD_DOUBLE * box_angle.y;
    double gamma = CONSTANT_DEG_TO_RAD_DOUBLE * box_angle.z;
    double za = std::sqrt(1 - cos(alpha) * cos(alpha) - cos(beta) * cos(beta) -
                          cos(gamma) * cos(gamma) +
                          2 * cos(alpha) * cos(beta) * cos(gamma));
    cell.a11 = a;
    cell.a21 = b * cos(gamma);
    cell.a22 = b * sin(gamma);
    cell.a31 = c * cos(beta);
    cell.a32 = c / sin(gamma) * (cos(alpha) - cos(beta) * cos(gamma));
    cell.a33 = c / sin(gamma) * za;
    return cell;
}
