#include "MC_barostat.h"

void MC_BAROSTAT_INFORMATION::Volume_Change_Attempt(VECTOR boxlength, float dt)
{
    if (CONTROLLER::MPI_rank == 0)
    {
        std::uniform_real_distribution<double> centered(-1.0, 1.0);
        double nrand = centered(generator);

        Delta_Box_Length = {0.0f, 0.0f, 0.0f};
        switch (couple_dimension)
        {
            case NO:
                if (only_direction > 0)
                    xyz = only_direction - 1;
                else
                    xyz = std::uniform_int_distribution<int>(0, 2)(generator);
                if (xyz == 0)
                {
                    Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                }
                else if (xyz == 1)
                {
                    Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                }
                else
                {
                    Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                }
                break;
            case XY:
                if (only_direction > 0)
                    xyz = only_direction - 1;
                else
                    xyz = std::uniform_int_distribution<int>(0, 1)(generator);
                if (xyz == 0)
                {
                    Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                }
                else
                {
                    Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                    Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                }
                break;
            case XZ:
                if (only_direction > 0)
                    xyz = only_direction - 1;
                else
                    xyz = std::uniform_int_distribution<int>(0, 1)(generator);
                if (xyz == 0)
                {
                    Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                }
                else
                {
                    Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                    Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                }
                break;
            case YZ:
                if (only_direction > 0)
                    xyz = only_direction - 1;
                else
                    xyz = std::uniform_int_distribution<int>(0, 1)(generator);
                if (xyz == 0)
                {
                    Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                }
                else
                {
                    Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                    Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                }
                break;
            case XYZ:
                xyz = 0;
                Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
                Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
                Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
                break;
        }

        New_Box_Length = boxlength + Delta_Box_Length;
        DeltaS = 0.0f;
        switch (couple_dimension)
        {
            case NO:
                break;
            case XY:
                if (xyz == 1)
                {
                    DeltaS = New_Box_Length.x * New_Box_Length.y -
                             boxlength.x * boxlength.y;
                }
                break;
            case XZ:
                if (xyz == 1)
                {
                    DeltaS = New_Box_Length.x * New_Box_Length.z -
                             boxlength.x * boxlength.z;
                }
                break;
            case YZ:
                if (xyz == 1)
                {
                    DeltaS = New_Box_Length.z * New_Box_Length.y -
                             boxlength.z * boxlength.y;
                }
                break;
            case XYZ:
                break;
        }
        double V = boxlength.x * boxlength.y * boxlength.z;
        newV = New_Box_Length.x * New_Box_Length.y * New_Box_Length.z;
        DeltaV = newV - V;
        VDevided = newV / V;
        VECTOR crd_scale_factor = New_Box_Length / boxlength;
        g = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        g.a11 = (crd_scale_factor.x - 1.0f) / dt;
        g.a22 = (crd_scale_factor.y - 1.0f) / dt;
        g.a33 = (crd_scale_factor.z - 1.0f) / dt;
    }
#ifdef USE_MPI
    MPI_Bcast(&g, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
}

int MC_BAROSTAT_INFORMATION::Check_MC_Barostat_Accept()
{
    total_count[xyz] += 1;
    float tmp_rand;
    if (CONTROLLER::MPI_rank == 0)
    {
        tmp_rand = std::uniform_real_distribution<float>(0.0f, 1.0f)(generator);
    }
#ifdef USE_MPI
    MPI_Bcast(&tmp_rand, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
    if (tmp_rand < accept_possibility)  // 接受
    {
        accept = 1;
        accep_count[xyz] += 1;
    }
    else
    {
        accept = 0;
    }
    return accept;
}

void MC_BAROSTAT_INFORMATION::Initial(CONTROLLER* controller, int atom_numbers,
                                      float target_pressure, VECTOR boxlength,
                                      LTMatrix3 cell, const char* module_name)
{
    controller->printf("START INITIALIZING MC BAROSTAT:\n");
    if (module_name == NULL)
    {
        strcpy(this->module_name, "monte_carlo_barostat");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (cell.a21 != 0.0f || cell.a31 != 0.0f || cell.a32 != 0.0f)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "MC_BAROSTAT_INFORMATION::Initial",
            "Reason:\n\t MC barostat only supports orthogonal box now.\n");
    }
    controller->printf("    The target pressure is %.2f bar\n",
                       target_pressure * CONSTANT_PRES_CONVERTION);
    float mc_baro_initial_ratio = 0.001;
    if (controller[0].Command_Exist(this->module_name, "initial_ratio"))
    {
        controller->Check_Float(this->module_name, "initial_ratio",
                                "MC_BAROSTAT_INFORMATION::Initial");
        mc_baro_initial_ratio =
            atof(controller[0].Command(this->module_name, "initial_ratio"));
    }
    Delta_Box_Length_Max[0] = mc_baro_initial_ratio * boxlength.x;
    Delta_Box_Length_Max[1] = mc_baro_initial_ratio * boxlength.y;
    Delta_Box_Length_Max[2] = mc_baro_initial_ratio * boxlength.z;
    controller->printf(
        "    The initial max box length to change is %f %f %f Angstrom for x y "
        "z\n",
        Delta_Box_Length_Max[0], Delta_Box_Length_Max[1],
        Delta_Box_Length_Max[2]);

    update_interval = 100;
    if (controller[0].Command_Exist(this->module_name, "update_interval"))
    {
        controller->Check_Int(this->module_name, "update_interval",
                              "MC_BAROSTAT_INFORMATION::Initial");
        update_interval =
            atoi(controller[0].Command(this->module_name, "update_interval"));
    }
    controller->printf("    The update_interval is %d\n", update_interval);

    check_interval = 10;
    if (controller[0].Command_Exist(this->module_name, "check_interval"))
    {
        controller->Check_Int(this->module_name, "check_interval",
                              "MC_BAROSTAT_INFORMATION::Initial");
        check_interval =
            atoi(controller[0].Command(this->module_name, "check_interval"));
    }
    controller->printf("    The check_interval is %d\n", check_interval);

    accept_rate_low = 30;
    if (controller[0].Command_Exist(this->module_name, "accept_rate_low"))
    {
        controller->Check_Float(this->module_name, "accept_rate_low",
                                "MC_BAROSTAT_INFORMATION::Initial");
        accept_rate_low =
            atof(controller[0].Command(this->module_name, "accept_rate_low"));
    }
    controller->printf("    The lowest accept rate is %.2f%%\n",
                       accept_rate_low);

    accept_rate_high = 40;
    if (controller[0].Command_Exist(this->module_name, "accept_rate_high"))
    {
        controller->Check_Float(this->module_name, "accept_rate_high",
                                "MC_BAROSTAT_INFORMATION::Initial");
        accept_rate_high =
            atof(controller[0].Command(this->module_name, "accept_rate_high"));
    }
    controller->printf("    The highest accept rate is %.2f%%\n",
                       accept_rate_high);

    if (!controller->Command_Exist(this->module_name, "couple_dimension") ||
        controller->Command_Choice(this->module_name, "couple_dimension",
                                   "XYZ"))
    {
        couple_dimension = XYZ;
    }
    else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "NO"))
    {
        couple_dimension = NO;
    }
    else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "XY"))
    {
        couple_dimension = XY;
    }
    else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "XZ"))
    {
        couple_dimension = XZ;
    }
    else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "YZ"))
    {
        couple_dimension = YZ;
    }
    if (!controller->Command_Exist(this->module_name, "couple_dimension"))
        controller->printf("    The couple dimension is %s (index %d)\n", "XYZ",
                           couple_dimension);
    else
        controller->printf(
            "    The couple dimension is %s (index %d)\n",
            controller->Command(this->module_name, "couple_dimension"),
            couple_dimension);
    if (controller->Command_Exist(this->module_name, "only_direction"))
    {
        if (couple_dimension == NO)
        {
            if (controller->Command_Choice(this->module_name, "only_direction",
                                           "x"))
            {
                only_direction = 1;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "y"))
            {
                only_direction = 2;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "z"))
            {
                only_direction = 3;
            }
        }
        else if (couple_dimension == XYZ)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand,
                "MC_BAROSTAT_INFORMATION::Initial",
                "Reason:\n\tonly_dimension is not valid for isotropic pressure "
                "regulation\n");
        }
        else if (couple_dimension == XY)
        {
            if (controller->Command_Choice(this->module_name, "only_direction",
                                           "z"))
            {
                only_direction = 1;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "xy"))
            {
                only_direction = 2;
            }
        }
        else if (couple_dimension == XZ)
        {
            if (controller->Command_Choice(this->module_name, "only_direction",
                                           "y"))
            {
                only_direction = 1;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "xz"))
            {
                only_direction = 2;
            }
        }
        else if (couple_dimension == YZ)
        {
            if (controller->Command_Choice(this->module_name, "only_direction",
                                           "x"))
            {
                only_direction = 1;
            }
            else if (controller->Command_Choice(this->module_name,
                                                "only_direction", "yz"))
            {
                only_direction = 2;
            }
        }
    }
    if (couple_dimension != NO && couple_dimension != XYZ)
    {
        surface_number = 0;
        if (controller->Command_Exist(this->module_name, "surface_number"))
        {
            controller->Check_Int(this->module_name, "surface_number",
                                  "MC_BAROSTAT_INFORMATION::Initial");
            surface_number =
                atoi(controller->Command(this->module_name, "surface_number"));
        }
        surface_tension = 0.0f;
        if (controller->Command_Exist(this->module_name, "surface_tension"))
        {
            controller->Check_Float(this->module_name, "surface_tension",
                                    "MC_BAROSTAT_INFORMATION::Initial");
            surface_tension =
                atof(controller->Command(this->module_name, "surface_tension"));
        }
        surface_tension *= TENSION_UNIT_FACTOR;
        controller->printf("        The surface number is %d\n",
                           surface_number);
        controller->printf("        The surface tension is %f\n",
                           surface_tension);
    }
    Device_Malloc_Safely((void**)&frc_backup, sizeof(VECTOR) * atom_numbers);
    Device_Malloc_Safely((void**)&crd_backup, sizeof(VECTOR) * atom_numbers);
    generator = std::default_random_engine(rand());
    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial("density", "%.4f");
        is_controller_printf_initialized = 1;
        controller[0].printf("    structure last modify date is %d\n",
                             last_modify_date);
    }
    controller[0].printf("END INITIALIZING MC BAROSTAT\n\n");
}

void MC_BAROSTAT_INFORMATION::Delta_Box_Length_Max_Update()
{
    if (total_count[xyz] % check_interval == 0)
    {
        accept_rate[xyz] = 100.0 * accep_count[xyz] / total_count[xyz];

        if (accept_rate[xyz] < accept_rate_low)
        {
            total_count[xyz] = 0;
            accep_count[xyz] = 0;
            Delta_Box_Length_Max[xyz] *= 0.9;
        }
        if (accept_rate[xyz] > accept_rate_high)
        {
            total_count[xyz] = 0;
            accep_count[xyz] = 0;
            Delta_Box_Length_Max[xyz] *= 1.1;
        }
    }
}

void MC_BAROSTAT_INFORMATION::Ask_For_Calculate_Potential(int steps,
                                                          int* need_potential)
{
    if (is_initialized && steps % update_interval == 0)
    {
        *need_potential = 1;
    }
}
